import pandas as pd
import pytest

import auth
import chat_store


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATACHAT_DB", str(tmp_path / "test.db"))


def make_user(name):
    auth.create_user(name, "password123")
    return auth.authenticate(name, "password123")["id"]


def sample_result():
    return pd.DataFrame({"City": ["Delhi", "Mumbai"], "Sales": [85000.5, 135000.0]})


def test_save_and_load_messages():
    user_id = make_user("rahul")
    session_id = chat_store.create_session(user_id, "Sales by city")

    chat_store.add_message(
        user_id, session_id, "sales by city?", "SELECT 1", "Mumbai leads", sample_result()
    )

    messages = chat_store.load_messages(user_id, session_id)

    assert messages[0]["question"] == "sales by city?"
    assert messages[0]["sql"] == "SELECT 1"
    assert messages[0]["result"]["City"].tolist() == ["Delhi", "Mumbai"]


def test_title_is_cleaned_and_shortened():
    user_id = make_user("rahul")
    chat_store.create_session(user_id, "  hello    world  " + "x" * 100)

    title = chat_store.list_sessions(user_id)[0]["title"]
    assert title.startswith("hello world") and len(title) <= 60


def test_users_cannot_see_each_others_chats():
    rahul = make_user("rahul")
    priya = make_user("priya")
    session_id = chat_store.create_session(rahul, "private")

    assert chat_store.list_sessions(priya) == []
    assert chat_store.load_messages(priya, session_id) == []

    with pytest.raises(ValueError):
        chat_store.add_message(priya, session_id, "q", "s", "e", sample_result())


def test_other_user_cannot_delete_a_chat():
    rahul = make_user("rahul")
    priya = make_user("priya")
    session_id = chat_store.create_session(rahul, "mine")

    chat_store.delete_session(priya, session_id)
    assert len(chat_store.list_sessions(rahul)) == 1

    chat_store.delete_session(rahul, session_id)
    assert chat_store.list_sessions(rahul) == []


def test_only_max_rows_are_stored(monkeypatch):
    monkeypatch.setattr(chat_store, "MAX_STORED_ROWS", 10)

    user_id = make_user("rahul")
    session_id = chat_store.create_session(user_id, "big")
    chat_store.add_message(
        user_id, session_id, "q", "s", "e", pd.DataFrame({"n": range(50)})
    )

    assert len(chat_store.load_messages(user_id, session_id)[0]["result"]) == 10
