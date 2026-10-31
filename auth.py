"""Sign-up and login. Passwords are never stored - only a salted PBKDF2 hash."""
import hashlib
import hmac
import os
import re
import sqlite3
from datetime import datetime, timezone

from app_db import db

ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 8
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS
    )
    return digest.hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Create an account. Returns (success, message)."""
    username = username.strip()

    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, underscore."

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."

    salt = os.urandom(16)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt.hex(), created_at),
            )
    except sqlite3.IntegrityError:
        return False, "This username is already taken."

    return True, "Account created. You can log in now."


def authenticate(username: str, password: str) -> dict | None:
    """Return {'id', 'username'} if the login is correct, otherwise None."""
    with db() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt FROM users "
            "WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()

    if row is None:
        # Do the same amount of work so timing does not reveal valid usernames.
        _hash_password(password, b"\x00" * 16)
        return None

    candidate = _hash_password(password, bytes.fromhex(row["salt"]))

    if hmac.compare_digest(candidate, row["password_hash"]):
        return {"id": row["id"], "username": row["username"]}

    return None
