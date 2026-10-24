import re

import duckdb
import pandas as pd


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
            return {base_name: next(iter(sheets.values()))}

        return {f"{base_name}_{sheet}": data for sheet, data in sheets.items()}

    try:
        dataframe = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        dataframe = pd.read_csv(uploaded_file, encoding="latin-1")

    return {base_name: dataframe}


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


def build_database(tables: dict):
    """Create an in-memory DuckDB database with one table per DataFrame."""
    connection = duckdb.connect(database=":memory:")

    for table_name, dataframe in tables.items():
        connection.register("temp_dataframe", dataframe)
        connection.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM temp_dataframe'
        )
        connection.unregister("temp_dataframe")

    return connection


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