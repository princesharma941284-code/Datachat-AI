"""Accuracy benchmark for DataChat AI.

Run from the project folder:

    python -m benchmark.run_benchmark              # real run (needs OPENAI_API_KEY)
    python -m benchmark.run_benchmark --mock       # no API call: checks the benchmark itself
    python -m benchmark.run_benchmark --limit 10   # quick try with 10 questions

For every question it:
  1. asks DataChat's own generate_sql() for a query,
  2. checks it with the same safety filter the app uses,
  3. runs it on the sample database,
  4. compares the result with the trusted gold query.

Output (folder benchmark/results by default):
  results.csv         one row per question
  summary.json        accuracy overall / per category / per difficulty
  accuracy_chart.png  the graph for the report and the PPT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from benchmark.evaluator import results_match
from benchmark.questions import CATEGORIES, QUESTIONS
from benchmark.sample_data import SAMPLE_DEFINITIONS, build_sample_tables
from db_manager import build_database, get_schema_text, run_query
from semantic_layer import apply_definitions, build_semantic_layer, semantic_schema_text
from sql_safety import validate_read_only_sql

DIFFICULTIES = ["easy", "medium", "hard"]


def evaluate_question(question: dict, connection, schema: str, generator) -> dict:
    """Run one benchmark question and return a result record."""
    record = {
        "id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "question": question["question"],
        "status": "",
        "correct": False,
        "predicted_sql": "",
        "gold_sql": question["gold_sql"],
        "error": "",
        "seconds": 0.0,
    }

    started = time.perf_counter()

    try:
        gold = run_query(connection, question["gold_sql"])

        try:
            raw_sql = generator(question["question"], schema)
        except Exception as error:  # API / network problem
            record.update(status="api_error", error=str(error)[:300])
            return record

        ok, result = validate_read_only_sql(raw_sql)
        record["predicted_sql"] = result if ok else raw_sql

        if not ok:
            record.update(status="blocked_by_safety", error=result)
            return record

        try:
            predicted = run_query(connection, result)
        except Exception as error:
            record.update(status="sql_error", error=str(error)[:300])
            return record

        if results_match(gold, predicted, ordered=question["ordered"]):
            record.update(status="correct", correct=True)
        else:
            record.update(status="wrong_result")

        return record
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)


def build_schema(tables: dict, connection, schema_mode: str) -> str:
    """"plain" = only table/column names (before Day 2).
    "semantic" = semantic layer: meanings, joins, metrics (Day 2)."""
    if schema_mode == "plain":
        return get_schema_text(connection, list(tables))

    layer = build_semantic_layer(tables)
    apply_definitions(layer, SAMPLE_DEFINITIONS)
    return semantic_schema_text(layer, connection)


def run_benchmark(
    questions,
    generator,
    delay: float = 0.0,
    progress=None,
    schema_mode: str = "semantic",
) -> pd.DataFrame:
    """Run all questions and return a DataFrame with one row per question."""
    tables = build_sample_tables()
    connection = build_database(tables)
    schema = build_schema(tables, connection, schema_mode)

    records = []
    for index, question in enumerate(questions, start=1):
        record = evaluate_question(question, connection, schema, generator)
        records.append(record)

        if progress:
            progress(index, len(questions), record)
        if delay:
            time.sleep(delay)

    return pd.DataFrame(records)


def _accuracy_by(results: pd.DataFrame, column: str, order: list[str]) -> dict:
    out = {}
    for name in order:
        part = results[results[column] == name]
        if len(part):
            out[name] = {
                "correct": int(part["correct"].sum()),
                "total": int(len(part)),
                "accuracy": round(100 * part["correct"].mean(), 1),
            }
    return out


def summarize(results: pd.DataFrame, model: str = "", schema_mode: str = "semantic") -> dict:
    total = len(results)
    correct = int(results["correct"].sum())
    return {
        "model": model,
        "schema_mode": schema_mode,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "total_questions": total,
        "correct": correct,
        "accuracy": round(100 * correct / total, 1) if total else 0.0,
        "by_category": _accuracy_by(results, "category", CATEGORIES),
        "by_difficulty": _accuracy_by(results, "difficulty", DIFFICULTIES),
        "status_counts": results["status"].value_counts().to_dict(),
        "avg_seconds_per_question": round(float(results["seconds"].mean()), 2) if total else 0.0,
    }


def make_chart(summary: dict, path) -> None:
    """Save the accuracy graph (bar charts by category and by difficulty)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    def draw(axis, data: dict, title: str, color: str):
        names = list(data)
        values = [data[n]["accuracy"] for n in names]
        bars = axis.bar(names, values, color=color)
        axis.set_ylim(0, 112)
        axis.set_ylabel("Accuracy (%)")
        axis.set_title(title)
        axis.axhline(summary["accuracy"], color="grey", linestyle="--", linewidth=1)
        axis.tick_params(axis="x", rotation=20)
        for bar, name in zip(bars, names):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%\n({data[name]['correct']}/{data[name]['total']})",
                ha="center", va="bottom", fontsize=9,
            )

    draw(left, summary["by_category"], "Accuracy by question type", "#4C78A8")
    draw(right, summary["by_difficulty"], "Accuracy by difficulty", "#59A14F")

    model = f" - {summary['model']}" if summary.get("model") else ""
    mode = {"plain": " (plain schema)", "semantic": " (semantic layer)"}.get(summary.get("schema_mode"), "")
    fig.suptitle(
        f"DataChat AI text-to-SQL accuracy{model}{mode}: "
        f"{summary['correct']}/{summary['total_questions']} correct "
        f"({summary['accuracy']}%)  [dashed line = overall]",
        fontsize=12,
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_comparison_chart(plain: dict, semantic: dict, path) -> None:
    """Before/after graph: plain schema vs semantic layer, per question type."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Overall"] + list(plain["by_category"])
    before = [plain["accuracy"]] + [plain["by_category"][n]["accuracy"] for n in plain["by_category"]]
    after = [semantic["accuracy"]] + [semantic["by_category"].get(n, {"accuracy": 0})["accuracy"]
                                      for n in plain["by_category"]]

    fig, axis = plt.subplots(figsize=(11, 5))
    positions = range(len(names))
    width = 0.38
    bars_before = axis.bar([p - width / 2 for p in positions], before, width, label="Plain schema (before)", color="#B0B7C3")
    bars_after = axis.bar([p + width / 2 for p in positions], after, width, label="With semantic layer (after)", color="#4C78A8")

    for bars in (bars_before, bars_after):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)

    axis.set_xticks(list(positions))
    axis.set_xticklabels(names, rotation=15)
    axis.set_ylim(0, 112)
    axis.set_ylabel("Accuracy (%)")
    model = f" - {semantic['model']}" if semantic.get("model") else ""
    axis.set_title(
        f"Effect of the semantic layer{model}: {plain['accuracy']}% -> {semantic['accuracy']}% "
        f"({semantic['total_questions']} questions)"
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_outputs(results: pd.DataFrame, summary: dict, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_chart(summary, out / "accuracy_chart.png")
    return out


def _mock_generator():
    """Pretend AI that always returns the gold SQL (tests the benchmark itself)."""
    gold = {q["question"]: q["gold_sql"] for q in QUESTIONS}
    return lambda question, schema: gold[question]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DataChat AI accuracy benchmark")
    parser.add_argument("--mock", action="store_true", help="use gold SQL instead of the real AI")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N questions")
    parser.add_argument("--category", choices=CATEGORIES, help="run one category only")
    parser.add_argument("--schema", choices=["plain", "semantic"], default="semantic",
                        help="what the AI sees: plain column list or the Day 2 semantic layer")
    parser.add_argument("--compare", action="store_true",
                        help="run BOTH plain and semantic and draw a before/after graph")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions")
    parser.add_argument("--out", default="benchmark/results", help="output folder")
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="exit with code 1 if accuracy (%%) is below this (for CI)")
    args = parser.parse_args(argv)

    questions = QUESTIONS
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    if args.mock:
        generator, model = _mock_generator(), "mock (gold SQL)"
    else:
        from llm_helper import generate_sql, get_model_name, has_api_key

        if not has_api_key():
            print("OPENAI_API_KEY is missing. Add it to .env (or use --mock).")
            return 2
        generator, model = generate_sql, get_model_name()

    def progress(done, total, record):
        mark = "OK  " if record["correct"] else "FAIL"
        print(f"[{done:>2}/{total}] {mark} {record['category']:<12} {record['question'][:60]}")

    if args.compare:
        runs = {}
        for mode in ("plain", "semantic"):
            print(f"\n=== schema: {mode} ===")
            results = run_benchmark(questions, generator, delay=args.delay,
                                    progress=progress, schema_mode=mode)
            runs[mode] = summarize(results, model, mode)
            save_outputs(results, runs[mode], Path(args.out) / mode)

        make_comparison_chart(runs["plain"], runs["semantic"], Path(args.out) / "comparison_chart.png")
        print(f"\nPlain schema:  {runs['plain']['accuracy']}%")
        print(f"Semantic layer: {runs['semantic']['accuracy']}%")
        print(f"\nSaved results to {args.out}/ (comparison_chart.png, plain/, semantic/)")
        summary = runs["semantic"]
    else:
        results = run_benchmark(questions, generator, delay=args.delay,
                                progress=progress, schema_mode=args.schema)
        summary = summarize(results, model, args.schema)
        out = save_outputs(results, summary, args.out)

        print(f"\nAccuracy: {summary['correct']}/{summary['total_questions']} = {summary['accuracy']}%")
        for name, item in summary["by_category"].items():
            print(f"  {name:<12} {item['correct']}/{item['total']}  ({item['accuracy']}%)")
        print(f"\nSaved results to {out}/")

    return 1 if summary["accuracy"] < args.min_accuracy else 0


if __name__ == "__main__":
    sys.exit(main())
