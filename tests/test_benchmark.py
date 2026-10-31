import pandas as pd
import pytest

pytest.importorskip("duckdb")

from benchmark.questions import CATEGORIES, QUESTIONS
from benchmark.run_benchmark import (
    _mock_generator,
    make_chart,
    make_comparison_chart,
    run_benchmark,
    save_outputs,
    summarize,
)
from benchmark.sample_data import build_sample_tables
from db_manager import build_database, run_query
from sql_safety import validate_read_only_sql


def test_there_are_50_questions_with_unique_ids_and_texts():
    assert len(QUESTIONS) == 50
    assert len({q["id"] for q in QUESTIONS}) == 50
    assert len({q["question"] for q in QUESTIONS}) == 50
    assert {q["category"] for q in QUESTIONS} == set(CATEGORIES)


def test_sample_data_is_the_same_every_time():
    first = build_sample_tables()["orders"]
    second = build_sample_tables()["orders"]
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: f"Q{q['id']}")
def test_gold_query_is_safe_and_returns_rows(question):
    ok, cleaned = validate_read_only_sql(question["gold_sql"])
    assert ok, cleaned

    connection = build_database(build_sample_tables())
    result = run_query(connection, cleaned)
    assert len(result) > 0


def test_perfect_ai_scores_100_percent():
    results = run_benchmark(QUESTIONS, _mock_generator())
    assert results["correct"].all(), results[~results["correct"]][["id", "status", "error"]]


def test_wrong_ai_scores_low_and_failures_are_classified():
    def bad_ai(question, schema):
        if question.startswith("How many orders are there"):
            return "SELECT COUNT(*) FROM orders WHERE 1 = 0"  # wrong answer
        if question.startswith("What is the total revenue?"):
            return "SELECT SUM(missing_column) FROM orders"  # SQL error
        if question.startswith("How many customers do we have"):
            return "DROP TABLE customers"  # blocked
        raise RuntimeError("API down")  # everything else

    results = run_benchmark(QUESTIONS[:6], bad_ai)
    statuses = results["status"].tolist()
    assert statuses[0] == "wrong_result"
    assert statuses[1] == "sql_error"
    assert statuses[4] == "blocked_by_safety"
    assert "api_error" in statuses
    assert not results["correct"].any()


def test_summary_and_chart_files_are_created(tmp_path):
    results = run_benchmark(QUESTIONS[:10], _mock_generator())
    summary = summarize(results, "mock")
    assert summary["accuracy"] == 100.0
    assert summary["total_questions"] == 10

    out = save_outputs(results, summary, tmp_path)
    assert (out / "results.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "accuracy_chart.png").stat().st_size > 1000


def test_plain_and_semantic_schema_are_really_different():
    seen = {}

    def spy(mode):
        def generator(question, schema):
            seen[mode] = schema
            return "SELECT 1"
        return generator

    run_benchmark(QUESTIONS[:1], spy("plain"), schema_mode="plain")
    run_benchmark(QUESTIONS[:1], spy("semantic"), schema_mode="semantic")

    assert "Business metrics" not in seen["plain"]
    assert "Business metrics" in seen["semantic"]
    assert "revenue = SUM(total_amount)" in seen["semantic"]
    assert "orders.customer_id = customers.customer_id" in seen["semantic"]
    assert "'Delivered'" in seen["semantic"]


def test_sample_metrics_formulas_work_on_the_sample_data():
    from benchmark.sample_data import SAMPLE_DEFINITIONS
    from semantic_layer import apply_definitions, build_semantic_layer, check_metrics

    tables = build_sample_tables()
    layer = apply_definitions(build_semantic_layer(tables), SAMPLE_DEFINITIONS)
    assert len(layer["metrics"]) == 2
    assert check_metrics(layer, build_database(tables)) == []


def test_comparison_chart_is_created(tmp_path):
    results = run_benchmark(QUESTIONS[:12], _mock_generator())
    plain = summarize(results, "mock", "plain")
    semantic = summarize(results, "mock", "semantic")

    make_comparison_chart(plain, semantic, tmp_path / "comparison.png")
    assert (tmp_path / "comparison.png").stat().st_size > 1000
