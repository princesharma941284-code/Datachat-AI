import pandas as pd

from benchmark.evaluator import results_match


def df(**columns):
    return pd.DataFrame(columns)


def test_same_numbers_match_even_with_tiny_rounding_difference():
    assert results_match(df(a=[12.34]), df(total=[12.3400001]))


def test_different_number_does_not_match():
    assert not results_match(df(a=[100]), df(a=[101]))


def test_column_names_do_not_matter():
    assert results_match(df(city=["Delhi"], n=[3]), df(x=["Delhi"], y=[3]))


def test_text_ignores_case_and_spaces():
    assert results_match(df(a=["Delhi"]), df(b=[" delhi "]))


def test_row_order_ignored_when_not_ordered():
    gold = df(a=["x", "y"], b=[1, 2])
    pred = df(a=["y", "x"], b=[2, 1])
    assert results_match(gold, pred, ordered=False)


def test_row_order_checked_when_ordered():
    gold = df(a=["x", "y"], b=[1, 2])
    pred = df(a=["y", "x"], b=[2, 1])
    assert not results_match(gold, pred, ordered=True)


def test_extra_columns_are_allowed():
    gold = df(name=["a", "b"])
    pred = df(id=[1, 2], name=["a", "b"])
    assert results_match(gold, pred)


def test_missing_column_fails():
    assert not results_match(df(a=[1], b=[2]), df(a=[1]))


def test_different_row_count_fails():
    assert not results_match(df(a=[1, 2]), df(a=[1]))


def test_columns_must_stay_paired_in_rows():
    gold = df(a=["x", "y"], b=[1, 2])
    pred = df(a=["x", "y"], b=[2, 1])  # same values, wrong pairing
    assert not results_match(gold, pred)


def test_dates_match_timestamp_and_text():
    gold = df(d=[pd.Timestamp("2026-01-05")])
    assert results_match(gold, df(d=["2026-01-05"]))


def test_null_values_match_each_other():
    assert results_match(df(a=[None, 1.0]), df(a=[1.0, None]))
