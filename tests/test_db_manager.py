import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_manager import make_table_name


def test_name_is_cleaned():
    assert make_table_name("Sample Sales.csv", set()) == "sample_sales_csv"


def test_name_starting_with_digit_gets_prefix():
    assert make_table_name("2026 data", set()) == "t_2026_data"


def test_duplicate_names_are_made_unique():
    used = set()
    first = make_table_name("sales", used)
    second = make_table_name("sales", used)
    assert first == "sales"
    assert second == "sales_2"


def test_empty_name_gets_default():
    assert make_table_name("!!!", set()) == "table"