import pytest

import auth


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATACHAT_DB", str(tmp_path / "test.db"))


def test_create_user_and_login():
    ok, _ = auth.create_user("rahul", "mypassword1")
    assert ok

    user = auth.authenticate("rahul", "mypassword1")
    assert user["username"] == "rahul"


def test_login_ignores_capital_letters_in_username():
    auth.create_user("Rahul", "mypassword1")
    assert auth.authenticate("RAHUL", "mypassword1") is not None


def test_wrong_password_fails():
    auth.create_user("rahul", "mypassword1")
    assert auth.authenticate("rahul", "wrongpass") is None


def test_unknown_user_fails():
    assert auth.authenticate("ghost", "whatever1") is None


def test_duplicate_username_is_rejected():
    auth.create_user("rahul", "mypassword1")
    ok, message = auth.create_user("RAHUL", "otherpass99")
    assert not ok and "taken" in message


def test_short_password_is_rejected():
    ok, _ = auth.create_user("rahul", "abc")
    assert not ok


def test_bad_username_is_rejected():
    ok, _ = auth.create_user("a b!", "longenough1")
    assert not ok


def test_password_is_not_stored_as_plain_text():
    from app_db import db

    auth.create_user("rahul", "mypassword1")

    with db() as conn:
        row = conn.execute("SELECT password_hash FROM users").fetchone()

    assert "mypassword1" not in row["password_hash"]
