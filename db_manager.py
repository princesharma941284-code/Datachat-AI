import re

import duckdb
import pandas as pd

DATE_PATTERN = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
)


def parse_date_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns that clearly hold dates (2026-01-10) into real dates.

    Without this DuckDB sees them as plain text and date questions
    ("sales by month") fail.
    """
    for column in dataframe.columns:
        series = dataframe[column]

        if not pd.api.types.is_string_dtype(series):
            continue

        values = series.dropna().astype(str).str.strip()

        if values.empty:
            continue

        if not values.head(50).str.match(DATE_PATTERN).all():
            continue

        parsed = pd.to_datetime(series, errors="coerce")

        if parsed.notna().sum() >= 0.9 * series.notna().sum():
            dataframe[column] = parsed

    return dataframe


def read_data_file(uploaded_file) -> dict:
    """Read a CSV or Excel file and return {label: DataFrame}.

    CSV  -> one entry.
    Excel -> one entry per sheet.
    """
    file_name = uploaded_file.name
    lower_name = file_name.lower()
    base_name = file_name.rsplit(".", 1)[0]

    if lower_name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(uploaded_file, sheet_name=None)

        if len(sheets) == 1:
            tables = {base_name: next(iter(sheets.values()))}
        else:
            tables = {f"{base_name}_{sheet}": data for sheet, data in sheets.items()}

    else:
        try:
            dataframe = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            dataframe = pd.read_csv(uploaded_file, encoding="latin-1")

        tables = {base_name: dataframe}

    return {label: parse_date_columns(data) for label, data in tables.items()}


def make_table_name(raw_name: str, used_names: set) -> str:
    """Convert a file/sheet name into a safe, unique SQL table name."""
    name = re.sub(r"\W+", "_", raw_name.strip().lower()).strip("_")

    if not name:
        name = "table"

    if name[0].isdigit():
        name = f"t_{name}"

    unique_name = name
    counter = 2

    while unique_name in used_names:
        unique_name = f"{name}_{counter}"
        counter += 1

    used_names.add(unique_name)
    return unique_name


def lock_down(connection) -> None:
    """Stop SQL from touching the computer's files or settings.

    This is the strong safety layer: even if a bad query slips past
    sql_safety.py, DuckDB itself refuses to read files (read_text,
    read_csv, ...) and the settings can no longer be changed.
    """
    connection.execute("SET enable_external_access = false")

    try:
        connection.execute("SET python_enable_replacements = false")
    except Exception:
        pass  # setting not available in this DuckDB version

    connection.execute("SET lock_configuration = true")


def build_database(tables: dict):
    """Create an in-memory DuckDB database with one table per DataFrame."""
    connection = duckdb.connect(database=":memory:")

    for table_name, dataframe in tables.items():
        connection.register("temp_dataframe", dataframe)
        connection.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM temp_dataframe'
        )
        connection.unregister("temp_dataframe")

    lock_down(connection)
    return connection


def run_query(connection, sql: str) -> pd.DataFrame:
    """Run one query on its own cursor (safe across Streamlit reruns)."""
    cursor = connection.cursor()

    try:
        return cursor.execute(sql).df()
    finally:
        cursor.close()


def get_schema_text(connection, table_names) -> str:
    """Describe every table in a compact text format for the AI prompt."""
    parts = []

    for table_name in table_names:
        schema = connection.execute(f'DESCRIBE "{table_name}"').df()
        columns = "\n".join(
            f"  - {row.column_name} ({row.column_type})"
            for row in schema.itertuples()
        )
        parts.append(f"Table: {table_name}\n{columns}")

    return "\n\n".join(parts)
