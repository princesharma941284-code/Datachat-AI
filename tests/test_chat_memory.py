from chat_memory import MAX_TURNS, add_turn, format_history_for_prompt


def test_empty_history_gives_empty_text():
    assert format_history_for_prompt([]) == ""


def test_add_turn_stores_all_parts():
    history = []
    add_turn(history, "q1", "SELECT 1", "answer", "result")

    assert history == [
        {"question": "q1", "sql": "SELECT 1", "explanation": "answer", "result": "result"}
    ]


def test_prompt_contains_question_and_sql():
    history = []
    add_turn(history, "total sales?", "SELECT SUM(x) FROM t", "ok", None)

    text = format_history_for_prompt(history)
    assert "total sales?" in text and "SELECT SUM(x) FROM t" in text


def test_only_last_turns_are_used():
    history = []
    for number in range(MAX_TURNS + 3):
        add_turn(history, f"question {number}", "SELECT 1", "ok", None)

    text = format_history_for_prompt(history)
    assert "question 0" not in text
    assert f"question {MAX_TURNS + 2}" in text
