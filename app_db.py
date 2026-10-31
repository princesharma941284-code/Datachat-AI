"""SQLite database for the app itself (users, chat sessions, messages).

This is separate from the DuckDB database that holds the user's uploaded data.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    sql         TEXT NOT NULL,
    explanation TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
"""


def get_db_path() -> str:
    """Where the SQLite file lives (override with DATACHAT_DB in .env)."""
    return os.getenv("DATACHAT_DB", "data/datachat.db")


def get_conn() -> sqlite3.Connection:
    """Open a connection and make sure all tables exist."""
    path = get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def db():
    """Use as:  with db() as conn: ...   (commits and closes automatically)"""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
