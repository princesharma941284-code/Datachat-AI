"""Day 2: automatic insights.

Right after upload we scan every table and write plain-language findings:
data quality, dominant categories, outliers, recent trend and strong
relations between numbers. We also suggest questions the user can click.

No AI call is needed for this (fast, free, same result every time).
ai_summary() can optionally turn the findings into a short written summary.
"""
from __future__ import annotations

import pandas as pd

from day3_features import detect_anomalies

MAX_INSIGHTS_PER_TABLE = 12

# Numbers like these should be averaged, not added up.
_RATE_LIKE = ("discount", "rate", "ratio", "percent", "pct", "price", "score", "age", "rating")
# Columns that usually hold the "main" money / quantity value.
_MONEY_LIKE = ("revenue", "sales", "amount", "total", "profit", "income", "cost")
_COUNT_LIKE = ("quantity", "qty", "units", "count")


def _is_rate_like(column: str) -> bool:
    return any(word in str(column).lower() for word in _RATE_LIKE)


def _rank_measures(measures: list[str]) -> list[str]:
    """Most important measures first (money / quantity columns before rates)."""
    def score(column: str) -> int:
        name = str(column).lower()
        if _is_rate_like(name):
            return 3
        if any(word in name for word in _MONEY_LIKE):
            return 0
        return 1 if any(word in name for word in _COUNT_LIKE) else 2
    return sorted(measures, key=score)


def _after_total(column: str) -> str:
    """'total_amount' -> 'amount', so we never write 'total total amount'."""
    text = _pretty(column)
    return text[6:] if text.lower().startswith("total ") else text


def _pretty(name: str) -> str:
    return str(name).replace("_", " ")


def _insight(table: str, kind: str, text: str, level: str = "info") -> dict:
    return {"table": table, "kind": kind, "level": level, "text": text}


def _fmt(value: float) -> str:
    value = float(value)
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------
# Insights per table
# ------------------------------------------------------------------

def _quality(table: str, frame: pd.DataFrame) -> list[dict]:
    found = []
    missing = frame.isna().mean() * 100
    for column, percent in missing[missing >= 5].sort_values(ascending=False).items():
        level = "warning" if percent >= 20 else "info"
        found.append(_insight(table, "quality", f"'{column}' has {percent:.0f}% missing values.", level))

    duplicates = int(frame.duplicated().sum())
    if duplicates:
        found.append(_insight(table, "quality", f"{duplicates} duplicate row(s) found.", "warning"))
    return found


def _categories(table: str, frame: pd.DataFrame, layer_table: dict) -> list[dict]:
    found = []
    for column, info in layer_table["columns"].items():
        if info["role"] != "category":
            continue
        counts = frame[column].value_counts(dropna=True)
        if counts.empty:
            continue
        if len(counts) < 2 or len(counts) > 10:
            continue
        share = 100 * counts.iloc[0] / counts.sum()
        level = "warning" if share >= 60 and len(counts) > 2 else "info"
        text = f"Most common {_pretty(column)}: {counts.index[0]} ({share:.0f}% of rows, {len(counts)} different values)."
        found.append(_insight(table, "category", text, level))
        if len(found) >= 3:
            break
    return found


def _measures(table: str, frame: pd.DataFrame, layer_table: dict) -> list[dict]:
    found = []
    measures = _rank_measures([c for c, i in layer_table["columns"].items() if i["role"] == "measure"])[:4]

    for column in measures:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue

        text = (
            f"{_pretty(column)}: total {_fmt(values.sum())}, average {_fmt(values.mean())}, "
            f"range {_fmt(values.min())} to {_fmt(values.max())}."
        )
        found.append(_insight(table, "measure", text))

        if len(values) >= 8:
            outliers = int(detect_anomalies(frame, column, 3.5)["_is_anomaly"].sum())
            if outliers:
                found.append(
                    _insight(table, "outlier", f"{outliers} unusual value(s) in '{column}' (far from the typical range).", "warning")
                )
    return found


def _trend(table: str, frame: pd.DataFrame, layer_table: dict) -> list[dict]:
    dates = [c for c, i in layer_table["columns"].items() if i["role"] == "date"]
    measures = _rank_measures([c for c, i in layer_table["columns"].items() if i["role"] == "measure"])
    if not dates or not measures:
        return []

    date_column = dates[0]
    stamps = pd.to_datetime(frame[date_column], errors="coerce")
    if stamps.notna().sum() < 6:
        return []

    span_days = (stamps.max() - stamps.min()).days
    freq, label = ("M", "month") if span_days >= 90 else ("W", "week")
    found = []

    for column in measures[:2]:
        data = pd.DataFrame({"d": stamps, "v": pd.to_numeric(frame[column], errors="coerce")}).dropna()
        grouped = data.groupby(data["d"].dt.to_period(freq))["v"]
        rate = _is_rate_like(column)
        series = grouped.mean() if rate else grouped.sum()
        if len(series) < 3:
            continue

        last, previous = series.iloc[-1], series.iloc[-2]
        best = series.idxmax()
        word = "Average" if rate else "Total"
        parts = [f"{word} {_after_total(column)} by {label}: highest in {best} ({_fmt(series.max())})."]

        if previous:
            change = 100 * (last - previous) / abs(previous)
            direction = "up" if change >= 0 else "down"
            parts.append(
                f"Latest {label} ({series.index[-1]}, may be incomplete) is {direction} {abs(change):.0f}% vs the {label} before."
            )
        found.append(_insight(table, "trend", " ".join(parts)))
    return found


def _correlations(table: str, frame: pd.DataFrame, layer_table: dict) -> list[dict]:
    measures = [c for c, i in layer_table["columns"].items() if i["role"] == "measure"]
    if len(measures) < 2 or len(frame) < 10:
        return []

    corr = frame[measures].apply(pd.to_numeric, errors="coerce").corr()
    found = []
    for i, first in enumerate(measures):
        for second in measures[i + 1:]:
            value = corr.loc[first, second]
            if pd.notna(value) and abs(value) >= 0.7:
                kind = "rise together" if value > 0 else "move in opposite directions"
                found.append(
                    _insight(table, "correlation", f"'{first}' and '{second}' {kind} (correlation {value:.2f}).")
                )
    return found[:3]


def generate_insights(tables: dict, layer: dict) -> list[dict]:
    """Return a list of {table, kind, level, text}. Never raises for bad columns."""
    insights = []

    for table_name, frame in tables.items():
        layer_table = layer["tables"][table_name]
        insights.append(
            _insight(table_name, "overview", f"{len(frame):,} rows and {len(frame.columns)} columns.")
        )

        collected = []
        for step in (_quality, _trend, _measures, _categories, _correlations):
            try:
                if step in (_quality,):
                    collected += step(table_name, frame)
                else:
                    collected += step(table_name, frame, layer_table)
            except Exception:
                continue  # one odd column must not hide all other insights

        collected.sort(key=lambda item: item["level"] != "warning")   # warnings first
        insights += collected[:MAX_INSIGHTS_PER_TABLE]

    return insights


# ------------------------------------------------------------------
# Suggested questions
# ------------------------------------------------------------------

def suggest_questions(tables: dict, layer: dict, limit: int = 8) -> list[str]:
    """Questions the user can click (built from column roles and metrics)."""
    questions = []

    for metric in layer["metrics"][:2]:
        questions.append(f"What is the total {metric['name']}?")

    for table_name, table in layer["tables"].items():
        columns = table["columns"]
        measures = _rank_measures([c for c, i in columns.items() if i["role"] == "measure"])
        categories = [c for c, i in columns.items() if i["role"] == "category"]
        dates = [c for c, i in columns.items() if i["role"] == "date"]

        if measures and categories:
            main = measures[0]
            if _is_rate_like(main):
                questions.append(f"What is the average {_pretty(main)} for each {_pretty(categories[0])}?")
            else:
                questions.append(f"What is the total {_after_total(main)} for each {_pretty(categories[0])}?")
                questions.append(f"Which {_pretty(categories[0])} has the highest average {_after_total(main)}?")
        if measures and dates:
            word = "average" if _is_rate_like(measures[0]) else "total"
            questions.append(f"How does the {word} {_after_total(measures[0])} change by month?")
        if not measures and categories:
            questions.append(f"How many rows are there for each {_pretty(categories[0])}?")

    for relation in layer["relationships"][:2]:
        child = layer["tables"][relation["from_table"]]["columns"]
        parent = layer["tables"][relation["to_table"]]["columns"]
        child_measures = _rank_measures([c for c, i in child.items() if i["role"] == "measure"])
        measure = child_measures[0] if child_measures and not _is_rate_like(child_measures[0]) else None
        category = next((c for c, i in parent.items() if i["role"] == "category"), None)
        if measure and category:
            questions.append(
                f"What is the total {_after_total(measure)} for each {_pretty(category)} (from {relation['to_table']})?"
            )

    unique = list(dict.fromkeys(questions))
    return unique[:limit]


# ------------------------------------------------------------------
# Optional AI summary
# ------------------------------------------------------------------

def insights_to_text(insights: list[dict]) -> str:
    return "\n".join(f"[{i['table']}] {i['text']}" for i in insights)


def ai_summary(insights: list[dict]) -> str:
    """Short written summary of the findings (uses the OpenAI API)."""
    from llm_helper import _generate_with_retry

    instructions = (
        "You are a business data analyst. Summarise the findings below for a "
        "manager in at most 5 short bullet points. Use only the given facts. "
        "Do not invent numbers or causes. Mention data problems first."
    )
    return _generate_with_retry(instructions, insights_to_text(insights))
