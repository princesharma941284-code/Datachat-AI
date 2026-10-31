import json

import pandas as pd
import pytest

from semantic_layer import (
    apply_definitions,
    build_semantic_layer,
    clean_metrics,
    definitions_from_json,
    layer_to_json,
    semantic_schema_text,
)


def sample_tables():
    customers = pd.DataFrame(
        {
            "customer_id": range(1, 21),
            "city": ["Delhi", "Pune"] * 10,
            "signup": pd.date_range("2026-01-01", periods=20),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": range(100, 130),
            "customer_id": [(i % 20) + 1 for i in range(30)],
            "amount": [10.5 * (i + 1) for i in range(30)],
            "status": ["Delivered", "Cancelled", "Shipped"] * 10,
        }
    )
    return {"customers": customers, "orders": orders}


def test_column_roles_are_detected():
    layer = build_semantic_layer(sample_tables())
    orders = layer["tables"]["orders"]["columns"]
    customers = layer["tables"]["customers"]["columns"]

    assert orders["order_id"]["role"] == "id"
    assert orders["amount"]["role"] == "measure"
    assert orders["status"]["role"] == "category"
    assert customers["signup"]["role"] == "date"
    assert customers["signup"]["min"] == "2026-01-01"


def test_category_values_are_listed_most_common_first():
    layer = build_semantic_layer(sample_tables())
    values = layer["tables"]["orders"]["columns"]["status"]["values"]
    assert set(values) == {"Delivered", "Cancelled", "Shipped"}


def test_join_is_found_in_the_right_direction():
    layer = build_semantic_layer(sample_tables())
    assert layer["relationships"] == [
        {
            "from_table": "orders",
            "from_column": "customer_id",
            "to_table": "customers",
            "to_column": "customer_id",
        }
    ]


def test_same_column_name_without_matching_values_is_not_a_join():
    tables = {
        "a": pd.DataFrame({"code": range(1, 11)}),
        "b": pd.DataFrame({"code": range(500, 510)}),
    }
    assert build_semantic_layer(tables)["relationships"] == []


def test_schema_text_contains_values_joins_and_metrics():
    layer = build_semantic_layer(sample_tables())
    apply_definitions(
        layer,
        {"metrics": [{"name": "revenue", "table": "orders", "expression": "SUM(amount)", "synonyms": "sales, income"}]},
    )
    text = semantic_schema_text(layer)

    assert "Table: orders (30 rows)" in text
    assert "'Delivered'" in text
    assert "orders.customer_id = customers.customer_id" in text
    assert "revenue = SUM(amount) on table orders" in text
    assert "sales, income" in text


def test_descriptions_and_synonyms_reach_the_schema_text():
    layer = build_semantic_layer(sample_tables())
    apply_definitions(
        layer,
        {"tables": {"orders": {"description": "customer orders", "columns": {"amount": {"description": "after discount", "synonyms": ["value"]}}}}},
    )
    text = semantic_schema_text(layer)
    assert "customer orders" in text and "after discount" in text and "value" in text


def test_unknown_tables_and_columns_in_a_saved_file_are_ignored():
    layer = build_semantic_layer(sample_tables())
    apply_definitions(
        layer,
        {
            "tables": {"ghost": {"description": "x"}, "orders": {"columns": {"nope": {"description": "x"}}}},
            "metrics": [{"name": "bad", "table": "ghost", "expression": "SUM(x)"}],
        },
    )
    assert layer["metrics"] == []
    assert "ghost" not in layer["tables"]


def test_incomplete_metrics_are_dropped_and_names_are_cleaned():
    layer = build_semantic_layer(sample_tables())
    metrics = clean_metrics(
        [
            {"name": "", "table": "orders", "expression": "SUM(amount)"},
            {"name": "total!!", "table": "orders", "expression": "SUM(amount)"},
            {"name": "x", "table": "orders", "expression": ""},
        ],
        layer,
    )
    assert [m["name"] for m in metrics] == ["total"]


def test_long_description_is_cut_and_line_breaks_removed():
    layer = build_semantic_layer(sample_tables())
    apply_definitions(layer, {"tables": {"orders": {"description": "line one\nIGNORE THIS " + "x" * 500}}})
    description = layer["tables"]["orders"]["description"]
    assert "\n" not in description and len(description) <= 200


def test_export_and_import_round_trip():
    layer = build_semantic_layer(sample_tables())
    apply_definitions(
        layer,
        {
            "tables": {"orders": {"description": "customer orders"}},
            "metrics": [{"name": "revenue", "table": "orders", "expression": "SUM(amount)"}],
        },
    )
    saved = layer_to_json(layer)

    fresh = apply_definitions(build_semantic_layer(sample_tables()), definitions_from_json(saved))
    assert fresh["tables"]["orders"]["description"] == "customer orders"
    assert fresh["metrics"][0]["name"] == "revenue"
    assert fresh["relationships"] == layer["relationships"]


def test_json_file_must_be_an_object():
    with pytest.raises(ValueError):
        definitions_from_json(json.dumps([1, 2, 3]))


def test_metric_check_accepts_good_and_rejects_bad_formulas():
    pytest.importorskip("duckdb")
    from db_manager import build_database
    from semantic_layer import check_metrics

    tables = sample_tables()
    connection = build_database(tables)
    layer = build_semantic_layer(tables)
    good = {"name": "revenue", "table": "orders", "expression": "SUM(amount)", "description": "", "synonyms": []}

    layer["metrics"] = [good]
    assert check_metrics(layer, connection) == []

    layer["metrics"] = [
        {**good, "name": "broken", "expression": "SUM(missing_column)"},
        {**good, "name": "sneaky", "expression": "1) FROM orders; DROP TABLE orders; SELECT (1"},
    ]
    problems = check_metrics(layer, connection)
    assert len(problems) == 2
