"""Day 2: semantic layer.

A semantic layer tells the AI what the data MEANS, not only what columns exist:
  - what each column is (id / date / category / number) with sample values
  - how tables connect (orders.customer_id = customers.customer_id)
  - business metrics, e.g.  revenue = SUM(total_amount)  (also "sales", "income")
  - short descriptions written by the user

It is built automatically from the uploaded tables (build_semantic_layer) and
the user can improve it (metrics, descriptions, relationships). The result is
turned into the schema text that is sent to the AI (semantic_schema_text).

This file only needs pandas. A DuckDB connection is optional and used only to
show exact column types and to test metric formulas.
"""
from __future__ import annotations

import json
import re

import pandas as pd

MAX_CATEGORY_VALUES = 20      # a text column with <= this many values is a "category"
MAX_TEXT_LENGTH = 200         # descriptions are cut to this length (keeps the prompt tidy)
# Identifier columns are recognised by NAME only. (Guessing from "all values
# are different" is wrong: a salary or amount column can also be all different.)
_ID_NAME = re.compile(r"(^|[_\s])(id|code|index|no|sno|serial)$|^unnamed", re.IGNORECASE)


def _clean_text(value, limit: int = MAX_TEXT_LENGTH) -> str:
    """One short line: no line breaks, no extra spaces."""
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _clean_synonyms(value) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    return [_clean_text(item, 40) for item in (value or []) if _clean_text(item, 40)]


# ------------------------------------------------------------------
# 1. Build the layer automatically
# ------------------------------------------------------------------

def _column_info(series: pd.Series, rows: int) -> dict:
    name = str(series.name)
    info = {"role": "text", "dtype": str(series.dtype), "description": "", "synonyms": []}
    non_null = series.dropna()

    if pd.api.types.is_datetime64_any_dtype(series):
        info["role"] = "date"
        if not non_null.empty:
            info["min"] = str(non_null.min().date())
            info["max"] = str(non_null.max().date())

    elif pd.api.types.is_bool_dtype(series):
        info["role"] = "category"
        info["values"] = [str(v) for v in non_null.unique()[:MAX_CATEGORY_VALUES]]

    elif pd.api.types.is_numeric_dtype(series):
        if _ID_NAME.search(name) or name.lower() == "id":
            info["role"] = "id"
        else:
            info["role"] = "measure"
            if not non_null.empty:
                convert = int if pd.api.types.is_integer_dtype(series) else (lambda v: round(float(v), 2))
                info["min"] = convert(non_null.min())
                info["max"] = convert(non_null.max())

    else:
        unique = non_null.astype(str).nunique()
        if 0 < unique <= MAX_CATEGORY_VALUES:
            info["role"] = "category"
            counts = non_null.astype(str).value_counts()
            info["values"] = counts.index.tolist()

    return info


def _singular(name: str) -> str:
    name = name.lower()
    return name[:-1] if name.endswith("s") else name


def _find_relationships(tables: dict) -> list[dict]:
    """Guess joins: same column name in two tables where one side is unique
    and (almost) all values of the other side exist in it."""
    found = []
    names = list(tables)

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for col_a in tables[first].columns:
                for col_b in tables[second].columns:
                    same_name = str(col_a).lower() == str(col_b).lower()
                    a_is_pk = str(col_a).lower() == "id" and str(col_b).lower() == f"{_singular(first)}_id"
                    b_is_pk = str(col_b).lower() == "id" and str(col_a).lower() == f"{_singular(second)}_id"

                    if not (same_name or a_is_pk or b_is_pk):
                        continue
                    if not (_id_like(tables[first][col_a]) and _id_like(tables[second][col_b])):
                        continue

                    relation = _orient(first, col_a, tables[first][col_a],
                                       second, col_b, tables[second][col_b])
                    if relation:
                        found.append(relation)

    return found


def _id_like(series: pd.Series) -> bool:
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    return not pd.api.types.is_bool_dtype(series)


def _orient(table_a, col_a, s_a, table_b, col_b, s_b):
    """Decide which side is the 'one' side (unique values) and which is 'many'."""
    def is_unique(series):
        non_null = series.dropna()
        return len(non_null) > 0 and non_null.is_unique

    def covered(child, parent):
        child_values = set(child.dropna().unique())
        return bool(child_values) and len(child_values & set(parent.dropna().unique())) / len(child_values) >= 0.8

    if is_unique(s_a) and covered(s_b, s_a):
        return {"from_table": table_b, "from_column": col_b, "to_table": table_a, "to_column": col_a}
    if is_unique(s_b) and covered(s_a, s_b):
        return {"from_table": table_a, "from_column": col_a, "to_table": table_b, "to_column": col_b}
    return None


def build_semantic_layer(tables: dict) -> dict:
    """Create a first draft of the semantic layer from DataFrames."""
    layer = {"tables": {}, "relationships": [], "metrics": []}

    for table_name, frame in tables.items():
        rows = len(frame)
        layer["tables"][table_name] = {
            "description": "",
            "rows": rows,
            "columns": {str(c): _column_info(frame[c], rows) for c in frame.columns},
        }

    layer["relationships"] = _find_relationships(tables)
    return layer


# ------------------------------------------------------------------
# 2. Let the user improve it
# ------------------------------------------------------------------

def apply_definitions(layer: dict, definitions: dict) -> dict:
    """Merge user definitions (descriptions, synonyms, metrics, relationships)
    into the layer. Unknown tables / columns are ignored, so an old JSON file
    never breaks a new upload. Returns the same layer object."""
    for table_name, table_def in (definitions.get("tables") or {}).items():
        table = layer["tables"].get(table_name)
        if not table:
            continue

        if table_def.get("description") is not None:
            table["description"] = _clean_text(table_def["description"])

        for column, column_def in (table_def.get("columns") or {}).items():
            info = table["columns"].get(column)
            if not info:
                continue
            if column_def.get("description") is not None:
                info["description"] = _clean_text(column_def["description"])
            if column_def.get("synonyms") is not None:
                info["synonyms"] = _clean_synonyms(column_def["synonyms"])

    if "metrics" in definitions:
        layer["metrics"] = clean_metrics(definitions["metrics"], layer)

    if definitions.get("relationships") is not None:
        layer["relationships"] = [
            r for r in definitions["relationships"] if _relationship_exists(layer, r)
        ]

    return layer


def _relationship_exists(layer: dict, relation: dict) -> bool:
    try:
        return (
            relation["from_column"] in layer["tables"][relation["from_table"]]["columns"]
            and relation["to_column"] in layer["tables"][relation["to_table"]]["columns"]
        )
    except (KeyError, TypeError):
        return False


def clean_metrics(metrics, layer: dict) -> list[dict]:
    """Keep only complete metrics that point at a real table."""
    cleaned = []
    for metric in metrics or []:
        name = re.sub(r"[^A-Za-z0-9_ ]+", "", str(metric.get("name", ""))).strip()
        expression = _clean_text(metric.get("expression", ""), 300)
        table = str(metric.get("table", "")).strip()

        if not name or not expression or table not in layer["tables"]:
            continue

        cleaned.append(
            {
                "name": name,
                "table": table,
                "expression": expression,
                "description": _clean_text(metric.get("description", "")),
                "synonyms": _clean_synonyms(metric.get("synonyms")),
            }
        )
    return cleaned


def check_metrics(layer: dict, connection) -> list[str]:
    """Run every metric formula once and return readable problems.

    Uses the same safety filter as the chat, so a metric can never be
    used to run anything except a read-only calculation.
    """
    from db_manager import run_query
    from sql_safety import validate_read_only_sql

    problems = []
    for metric in layer["metrics"]:
        sql = f'SELECT {metric["expression"]} AS value FROM "{metric["table"]}"'
        ok, checked = validate_read_only_sql(sql)

        if not ok:
            problems.append(f'{metric["name"]}: {checked}')
            continue
        try:
            run_query(connection, checked)
        except Exception as error:
            problems.append(f'{metric["name"]}: formula does not work ({str(error).splitlines()[0][:120]})')

    return problems


def layer_to_json(layer: dict) -> str:
    """Export only what a human wrote or can edit (not the auto-detected facts)."""
    tables = {}
    for table_name, table in layer["tables"].items():
        columns = {
            c: {k: v for k, v in (("description", i["description"]), ("synonyms", i["synonyms"])) if v}
            for c, i in table["columns"].items()
            if i["description"] or i["synonyms"]
        }
        if table["description"] or columns:
            tables[table_name] = {k: v for k, v in (("description", table["description"]), ("columns", columns)) if v}

    return json.dumps(
        {"tables": tables, "metrics": layer["metrics"], "relationships": layer["relationships"]},
        indent=2,
        ensure_ascii=False,
    )


def definitions_from_json(text: str) -> dict:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("The file must contain a JSON object.")
    return data


# ------------------------------------------------------------------
# 3. Turn the layer into the text the AI reads
# ------------------------------------------------------------------

def _describe_column(name: str, info: dict, sql_type: str) -> str:
    role = info["role"]
    if role == "date" and "min" in info:
        detail = f"date, {info['min']} to {info['max']}"
    elif role == "category" and info.get("values"):
        detail = "category, values: " + ", ".join(f"'{v}'" for v in info["values"])
    elif role == "measure" and "min" in info:
        detail = f"number, {info['min']} to {info['max']}"
    else:
        detail = {"id": "identifier", "measure": "number", "text": "text"}.get(role, role)

    line = f"  - {name} ({sql_type}) [{detail}]"
    if info["description"]:
        line += f" - {info['description']}"
    if info["synonyms"]:
        line += f" (also called: {', '.join(info['synonyms'])})"
    return line


def _sql_types(connection, table_name: str, layer: dict) -> dict:
    if connection is not None:
        try:
            frame = connection.execute(f'DESCRIBE "{table_name}"').df()
            return dict(zip(frame["column_name"], frame["column_type"]))
        except Exception:
            pass
    return {c: i["dtype"] for c, i in layer["tables"][table_name]["columns"].items()}


def semantic_schema_text(layer: dict, connection=None) -> str:
    """The schema text sent to the AI (replaces the plain column list)."""
    parts = []

    for table_name, table in layer["tables"].items():
        types = _sql_types(connection, table_name, layer)
        header = f"Table: {table_name} ({table['rows']} rows)"
        if table["description"]:
            header += f" - {table['description']}"

        lines = [header]
        lines += [_describe_column(c, i, types.get(c, i["dtype"])) for c, i in table["columns"].items()]
        parts.append("\n".join(lines))

    if layer["relationships"]:
        lines = ["Relationships (use these for JOINs):"]
        lines += [
            f"  - {r['from_table']}.{r['from_column']} = {r['to_table']}.{r['to_column']}"
            for r in layer["relationships"]
        ]
        parts.append("\n".join(lines))

    if layer["metrics"]:
        lines = ["Business metrics (when the question uses one of these words, use this exact formula):"]
        for m in layer["metrics"]:
            line = f"  - {m['name']} = {m['expression']} on table {m['table']}"
            if m["synonyms"]:
                line += f" (also called: {', '.join(m['synonyms'])})"
            if m["description"]:
                line += f" - {m['description']}"
            lines.append(line)
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
