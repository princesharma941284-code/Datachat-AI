import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_safety import clean_sql, validate_read_only_sql


def test_simple_select_is_allowed():
    ok, query = validate_read_only_sql("SELECT * FROM sales")
    assert ok is True
    assert query == "SELECT * FROM sales"


def test_with_query_is_allowed():
    ok, _ = validate_read_only_sql("WITH t AS (SELECT 1) SELECT * FROM t")
    assert ok is True


def test_join_query_is_allowed():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    ok, _ = validate_read_only_sql(sql)
    assert ok is True


def test_trailing_semicolon_is_removed():
    ok, query = validate_read_only_sql("SELECT 1;")
    assert ok is True
    assert query == "SELECT 1"


def test_markdown_code_fence_is_cleaned():
    assert clean_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_drop_is_blocked():
    ok, _ = validate_read_only_sql("DROP TABLE sales")
    assert ok is False


def test_delete_is_blocked():
    ok, _ = validate_read_only_sql("DELETE FROM sales")
    assert ok is False


def test_hidden_drop_inside_select_is_blocked():
    ok, _ = validate_read_only_sql("SELECT 1; DROP TABLE sales;")
    assert ok is False


def test_multiple_queries_are_blocked():
    ok, _ = validate_read_only_sql("SELECT 1; SELECT 2;")
    assert ok is False


def test_empty_sql_is_blocked():
    ok, _ = validate_read_only_sql("   ")
    assert ok is False