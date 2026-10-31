"""Compare an AI-generated query result with the trusted (gold) result.

This is "execution accuracy": we do not compare the SQL text, we compare
what the two queries actually return. Different SQL that returns the same
answer is correct.

Rules:
- Column names do not matter, only the values.
- Numbers are rounded to 2 decimals (so 12.3400001 == 12.34).
- Text is compared ignoring capital letters and extra spaces.
- Extra columns in the AI result are allowed (e.g. it also shows an id).
- If ordered=True the row order must match, otherwise any order is fine.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd


def _norm_value(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating, Decimal)):
        return round(float(value), 2)
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return str(pd.Timestamp(value).date())
    return str(value).strip().lower()


def _columns(frame: pd.DataFrame) -> list[list]:
    return [[_norm_value(v) for v in frame.iloc[:, i]] for i in range(frame.shape[1])]


def _sort_key(value):
    return (value is None, str(value))


def _rows(columns: list[list]) -> list[tuple]:
    return list(zip(*columns)) if columns else []


def _mappings(gold_cols, pred_cols):
    """Yield every way to pick, for each gold column, a matching AI column."""
    pred_sorted = [sorted(c, key=_sort_key) for c in pred_cols]
    gold_sorted = [sorted(c, key=_sort_key) for c in gold_cols]

    def walk(g_index, used):
        if g_index == len(gold_cols):
            yield []
            return
        for p_index, candidate in enumerate(pred_sorted):
            if p_index in used or candidate != gold_sorted[g_index]:
                continue
            for rest in walk(g_index + 1, used | {p_index}):
                yield [p_index] + rest

    yield from walk(0, frozenset())


def results_match(gold: pd.DataFrame, predicted: pd.DataFrame, ordered: bool = False) -> bool:
    """True when the AI result gives the same answer as the gold result."""
    if len(gold) != len(predicted):
        return False

    gold_cols = _columns(gold)
    pred_cols = _columns(predicted)

    if len(pred_cols) < len(gold_cols):
        return False

    gold_rows = _rows(gold_cols)
    if not ordered:
        gold_rows = sorted(gold_rows, key=lambda r: [_sort_key(v) for v in r])

    for mapping in _mappings(gold_cols, pred_cols):
        rows = _rows([pred_cols[i] for i in mapping])
        if not ordered:
            rows = sorted(rows, key=lambda r: [_sort_key(v) for v in r])
        if rows == gold_rows:
            return True

    return False
