import pytest

from sql_safety import validate_read_only_sql


def test_plain_select_is_allowed():
    ok, query = validate_read_only_sql("SELECT * FROM sales")
    assert ok and query == "SELECT * FROM sales"


def test_trailing_semicolon_is_removed():
    ok, query = validate_read_only_sql("SELECT 1;")
    assert ok and query == "SELECT 1"


def test_markdown_fences_are_removed():
    ok, query = validate_read_only_sql("```sql\nSELECT 1\n```")
    assert ok and query == "SELECT 1"


def test_with_query_is_allowed():
    ok, _ = validate_read_only_sql("WITH t AS (SELECT 1 AS a) SELECT * FROM t")
    assert ok


def test_join_is_allowed():
    ok, _ = validate_read_only_sql(
        "SELECT * FROM a JOIN b ON a.id = b.id"
    )
    assert ok


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT REPLACE(City, 'Delhi', 'New Delhi') FROM sales",
        "SELECT * REPLACE (Sales / 100 AS Sales) FROM sales",
        "SELECT * FROM tickets WHERE status = 'set'",
        "SELECT * FROM logs WHERE msg LIKE '%update%'",
        'SELECT "Load" FROM machines',
        "SELECT * FROM notes WHERE txt = 'a;b'",
        "SELECT read_count, last_update FROM stats",
        "SELECT 'it''s; DROP TABLE x' AS t",
    ],
)
def test_legit_queries_are_not_blocked(sql):
    ok, message = validate_read_only_sql(sql)
    assert ok, message


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "SELECT 1; DROP TABLE sales;",
        "SELECT 1; SELECT 2",
        "WITH t AS (SELECT 1) INSERT INTO sales SELECT * FROM t",
        "CREATE OR REPLACE TABLE x AS SELECT 1",
        "COPY sales TO 'x.csv'",
        "SELECT 1 /* x */; DELETE FROM sales",
        "SELECT $$a$$",
        "",
        "   ",
    ],
)
def test_unsafe_queries_are_blocked(sql):
    ok, _ = validate_read_only_sql(sql)
    assert not ok


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT content FROM read_text('.env')",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM glob('*')",
        "SELECT * FROM '../.env'",
        "SELECT * FROM duckdb_settings()",
        "SELECT * FROM query('DROP TABLE sales')",
    ],
)
def test_file_and_system_access_is_blocked(sql):
    ok, _ = validate_read_only_sql(sql)
    assert not ok
