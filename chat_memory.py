MAX_TURNS = 10


def add_turn(history, question, sql, explanation, result):
    history.append(
        {
            "question": question,
            "sql": sql,
            "explanation": explanation,
            "result": result,
        }
    )
    del history[:-MAX_TURNS]


def format_history_for_prompt(history):
    if not history:
        return ""

    parts = []
    for turn in history[-MAX_TURNS:]:
        parts.append(
            f"Question: {turn['question']}\n"
            f"SQL: {turn['sql']}\n"
            f"Answer: {turn['explanation']}"
        )
    return "\n\n".join(parts)
