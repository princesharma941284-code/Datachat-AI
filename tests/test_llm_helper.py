import pandas as pd
import pytest

pytest.importorskip("openai")

import llm_helper


class FakeResponse:
    def __init__(self, text):
        self.output_text = text


class FakeClient:
    """Fails with the given errors first, then answers."""

    def __init__(self, errors, answer="  SELECT 1  "):
        self.errors = list(errors)
        self.answer = answer
        self.calls = 0
        self.responses = self

    def create(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return FakeResponse(self.answer)


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_helper.time, "sleep", lambda seconds: None)


def use_client(monkeypatch, client):
    monkeypatch.setattr(llm_helper, "get_client", lambda: client)


def test_temporary_errors_are_retried_then_succeed(monkeypatch):
    client = FakeClient([StatusError(503), StatusError(429)])
    use_client(monkeypatch, client)

    assert llm_helper._generate_with_retry("rules", "question") == "SELECT 1"
    assert client.calls == 3


def test_permanent_errors_fail_immediately(monkeypatch):
    client = FakeClient([StatusError(401)])
    use_client(monkeypatch, client)

    with pytest.raises(StatusError):
        llm_helper._generate_with_retry("rules", "question")
    assert client.calls == 1


def test_gives_up_after_all_attempts(monkeypatch):
    client = FakeClient([StatusError(503)] * 10)
    use_client(monkeypatch, client)

    with pytest.raises(StatusError):
        llm_helper._generate_with_retry("rules", "question", attempts=3)
    assert client.calls == 3


def test_is_transient_uses_status_code():
    assert llm_helper._is_transient(StatusError(500))
    assert not llm_helper._is_transient(StatusError(400))


def test_generate_sql_puts_schema_and_history_in_prompt(monkeypatch):
    seen = {}

    def fake(instructions, user_input, attempts=4):
        seen["input"] = user_input
        return "SELECT 1"

    monkeypatch.setattr(llm_helper, "_generate_with_retry", fake)
    llm_helper.generate_sql("how many?", "Table: sales", history="Question: before")

    assert "Table: sales" in seen["input"]
    assert "Question: before" in seen["input"]
    assert "how many?" in seen["input"]


def test_explain_results_skips_api_when_result_is_empty(monkeypatch):
    monkeypatch.setattr(
        llm_helper, "_generate_with_retry",
        lambda *a, **k: pytest.fail("API must not be called for empty results"),
    )
    assert "No matching data" in llm_helper.explain_results("q", pd.DataFrame())


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_helper.has_api_key() is False
