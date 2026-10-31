import pandas as pd

from auto_insights import generate_insights, suggest_questions
from semantic_layer import apply_definitions, build_semantic_layer


def make_tables():
    days = pd.date_range("2026-01-01", periods=120)
    sales = pd.DataFrame(
        {
            "sale_id": range(1, 121),
            "day": days,
            "region": ["North"] * 80 + ["South"] * 40,
            "amount": [100 + i for i in range(120)],
            "units": [1 + i // 10 for i in range(120)],
            "discount": [0.1] * 120,
            "notes": [None] * 40 + ["ok"] * 80,
        }
    )
    sales.loc[5, "amount"] = 100000  # one clear outlier
    return {"sales": sales}


def insights_text(tables=None):
    tables = tables or make_tables()
    layer = build_semantic_layer(tables)
    return generate_insights(tables, layer)


def test_every_table_gets_an_overview():
    found = insights_text()
    assert any(i["kind"] == "overview" and "120 rows" in i["text"] for i in found)


def test_missing_values_are_reported_as_warning():
    found = [i for i in insights_text() if i["kind"] == "quality"]
    assert any("notes" in i["text"] and "33%" in i["text"] and i["level"] == "warning" for i in found)


def test_duplicate_rows_are_reported():
    frame = pd.DataFrame({"a": [1, 1, 2, 3, 4, 5, 6, 7], "b": list("xxyzabcd")})
    found = [i for i in insights_text({"t": frame}) if i["kind"] == "quality"]
    assert any("duplicate" in i["text"] for i in found)


def test_outlier_is_reported():
    found = [i for i in insights_text() if i["kind"] == "outlier"]
    assert any("amount" in i["text"] for i in found)


def test_dominant_category_is_reported():
    found = [i for i in insights_text() if i["kind"] == "category"]
    assert any("North" in i["text"] and "67%" in i["text"] for i in found)


def test_trend_mentions_latest_period():
    found = [i for i in insights_text() if i["kind"] == "trend"]
    assert found and "Latest month" in found[0]["text"]


def test_rate_columns_are_averaged_not_summed():
    found = [i["text"] for i in insights_text() if i["kind"] == "trend"]
    assert not any(text.startswith("Total discount") for text in found)


def test_strong_correlation_is_found():
    frame = pd.DataFrame({"x": range(30), "y": [v * 2 + 1 for v in range(30)], "z": [5, 1, 9, 2] * 7 + [3, 3]})
    found = [i for i in insights_text({"t": frame}) if i["kind"] == "correlation"]
    assert any("'x' and 'y'" in i["text"] for i in found)


def test_warnings_come_before_info():
    found = [i for i in insights_text() if i["kind"] != "overview"]
    levels = [i["level"] for i in found]
    assert levels == sorted(levels, key=lambda level: level != "warning")


def test_weird_data_does_not_crash():
    frame = pd.DataFrame({"a": [None] * 5, "b": ["x", "y", "z", "w", "v"]})
    assert insights_text({"t": frame})  # at least the overview


def test_suggested_questions_use_metrics_and_real_columns():
    tables = make_tables()
    layer = build_semantic_layer(tables)
    apply_definitions(layer, {"metrics": [{"name": "revenue", "table": "sales", "expression": "SUM(amount)"}]})

    questions = suggest_questions(tables, layer)
    assert questions[0] == "What is the total revenue?"
    assert any("amount" in q and "region" in q for q in questions)
    assert len(questions) == len(set(questions)) <= 8
