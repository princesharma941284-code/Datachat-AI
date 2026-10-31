"""Save and load chat sessions so they survive page refresh and logout."""
import io
import os
from datetime import datetime, timezone

import pandas as pd

from app_db import db

# Rows saved per answer. Raise it in .env with MAX_STORED_ROWS=5000
# (bigger = more disk space used and slower chat loading).
MAX_STORED_ROWS = int(os.getenv("MAX_STORED_ROWS", "1000"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(user_id: int, title: str) -> int:
    """Start a new chat and return its id."""
    title = " ".join(title.split())[:60] or "New chat"
    now = _now()

    with db() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_sessions (user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
        return cursor.lastrowid


def list_sessions(user_id: int) -> list[dict]:
    """All chats of this user, newest first."""
    with db() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM chat_sessions "
            "WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_session(user_id: int, session_id: int) -> None:
    """Delete one chat (only if it belongs to this user)."""
    with db() as conn:
        conn.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )


def add_message(
    user_id: int,
    session_id: int,
    question: str,
    sql: str,
    explanation: str,
    result: pd.DataFrame,
) -> None:
    """Save one question + answer into a chat."""
    result_json = result.head(MAX_STORED_ROWS).to_json(
        orient="split", date_format="iso"
    )
    now = _now()

    with db() as conn:
        owned = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()

        if owned is None:
            raise ValueError("This chat does not belong to the current user.")

        conn.execute(
            "INSERT INTO chat_messages "
            "(session_id, question, sql, explanation, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, question, sql, explanation, result_json, now),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )


def load_messages(user_id: int, session_id: int) -> list[dict]:
    """Messages of a chat, in the same shape chat_memory.add_turn() uses."""
    with db() as conn:
        rows = conn.execute(
            "SELECT m.question, m.sql, m.explanation, m.result_json "
            "FROM chat_messages m "
            "JOIN chat_sessions s ON s.id = m.session_id "
            "WHERE s.id = ? AND s.user_id = ? "
            "ORDER BY m.id",
            (session_id, user_id),
        ).fetchall()

    return [
        {
            "question": row["question"],
            "sql": row["sql"],
            "explanation": row["explanation"],
            "result": pd.read_json(io.StringIO(row["result_json"]), orient="split"),
        }
        for row in rows
    ]
