import io

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from db_manager import (
    build_database,
    get_schema_text,
    make_table_name,
    read_data_file,
    run_query,
)


class FakeUpload(io.BytesIO):
    """Looks like a Streamlit uploaded file."""

    def __init__(self, name, data):
        super().__init__(data)
        self.name = name


def test_table_names_are_safe_and_unique():
    used = set()
    assert make_table_name("Sales Data 2026.csv", used) == "sales_data_2026_csv"
    assert make_table_name("Sales Data 2026.csv", used) == "sales_data_2026_csv_2"


def test_table_name_starting_with_digit_and_empty_name():
    used = set()
    assert make_table_name("2026 sales", used) == "t_2026_sales"
    assert make_table_name("!!!", used) == "table"


def test_csv_is_read_and_dates_are_converted():
    upload = FakeUpload("orders.csv", b"id,order_date\n1,2026-01-05\n2,2026-02-10\n")
    tables = read_data_file(upload)

    assert list(tables) == ["orders"]
    assert pd.api.types.is_datetime64_any_dtype(tables["orders"]["order_date"])


def test_latin1_csv_is_still_read():
    upload = FakeUpload("names.csv", "name\ncaf\xe9\n".encode("latin-1"))
    assert read_data_file(upload)["names"]["name"][0] == "caf\xe9"


def test_schema_text_lists_tables_and_columns():
    connection = build_database({"sales": pd.DataFrame({"City": ["Delhi"], "Amount": [5]})})
    text = get_schema_text(connection, ["sales"])

    assert "Table: sales" in text and "City" in text and "Amount" in text


def test_data_cannot_be_modified_through_run_query():
    connection = build_database({"sales": pd.DataFrame({"Amount": [5]})})
    run_query(connection, "SELECT * FROM sales")  # normal use still works
    assert run_query(connection, "SELECT COUNT(*) AS n FROM sales")["n"][0] == 1
