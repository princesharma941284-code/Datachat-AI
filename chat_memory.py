MAX_TURNS = 5


def add_turn(history: list, question: str, sql: str, explanation: str, result):
    """Add one question-answer turn to the chat history list."""
    history.append(
        {
            "question": question,
            "sql": sql,
            "explanation": explanation,
            "result": result,
        }
    )


def format_history_for_prompt(history: list, max_turns: int = MAX_TURNS) -> str:
    """Turn the last few turns into short text so the AI can understand follow-ups."""
    if not history:
        return ""

    lines = []

    for turn in history[-max_turns:]:
        lines.append(f"User asked: {turn['question']}")
        lines.append(f"SQL used: {turn['sql']}")

    return "\n".join(lines)