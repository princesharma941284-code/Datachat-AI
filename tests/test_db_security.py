import pandas as pd
import pytest

pytest.importorskip("duckdb")

from db_manager import build_database, parse_date_columns, run_query


def make_connection():
    return build_database(
        {"sales": pd.DataFrame({"City": ["Delhi", "Mumbai"], "Sales": [10, 20]})}
    )


def test_normal_query_still_works():
    result = run_query(make_connection(), "SELECT SUM(Sales) AS total FROM sales")
    assert result["total"][0] == 30


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM read_csv('some_file.csv')",
        "SELECT * FROM glob('*')",
    ],
)
def test_duckdb_itself_refuses_file_access(sql):
    with pytest.raises(Exception):
        run_query(make_connection(), sql)


def test_settings_cannot_be_changed_after_lock():
    with pytest.raises(Exception):
        run_query(make_connection(), "SET enable_external_access = true")


def test_date_text_becomes_real_dates():
    frame = pd.DataFrame(
        {"OrderDate": ["2026-01-10", "2026-02-11"], "Name": ["a", "b"]}
    )
    frame = parse_date_columns(frame)

    assert str(frame["OrderDate"].dtype).startswith("datetime64")
    assert not str(frame["Name"].dtype).startswith("datetime64")


def test_normal_text_is_not_turned_into_dates():
    frame = pd.DataFrame({"Code": ["A-1", "B-2"]})
    assert not str(parse_date_columns(frame)["Code"].dtype).startswith("datetime64")
