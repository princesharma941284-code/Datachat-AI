import smtplib

import pandas as pd
import pytest

import day3_features
from day3_features import email_is_configured, parse_recipients, send_email_report


class FakeSMTP:
    """Replaces smtplib.SMTP so no real email is sent."""

    sent = []
    login_error = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ehlo(self):
        pass

    def starttls(self, context=None):
        pass

    def login(self, username, password):
        if FakeSMTP.login_error:
            raise FakeSMTP.login_error

    def send_message(self, message):
        FakeSMTP.sent.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.sent = []
    FakeSMTP.login_error = None
    monkeypatch.setattr(day3_features.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "abcdabcdabcdabcd")
    monkeypatch.delenv("SMTP_FROM", raising=False)
    monkeypatch.delenv("SMTP_USE_TLS", raising=False)
    return FakeSMTP


def test_single_and_multiple_recipients_are_parsed():
    assert parse_recipients("a@x.com") == ["a@x.com"]
    assert parse_recipients("a@x.com, b@y.org;  c@z.in\nd@w.com") == [
        "a@x.com", "b@y.org", "c@z.in", "d@w.com"
    ]


def test_duplicates_are_removed_ignoring_case():
    assert parse_recipients("A@x.com, a@x.com") == ["A@x.com"]


@pytest.mark.parametrize("bad", ["", "   ", "not-an-email", "a@b", "Bob <bob@x.com>"])
def test_bad_recipients_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_recipients(bad)


def test_too_many_recipients_are_rejected():
    many = ",".join(f"user{i}@x.com" for i in range(11))
    with pytest.raises(ValueError):
        parse_recipients(many)


def test_email_is_sent_with_csv_attachment(smtp):
    frame = pd.DataFrame({"city": ["Delhi"], "total": [10]})
    send_email_report("a@x.com, b@y.com", "Sales", "report body", frame, "my result.csv")

    message = smtp.sent[0]
    assert message["To"] == "a@x.com, b@y.com"
    assert message["From"] == "me@gmail.com"
    assert message["Subject"] == "Sales"

    attachment = list(message.iter_attachments())[0]
    assert attachment.get_filename() == "my_result.csv"
    assert "Delhi" in attachment.get_content()


def test_email_without_attachment(smtp):
    send_email_report("a@x.com", "Hi", "body")
    assert list(smtp.sent[0].iter_attachments()) == []


def test_line_breaks_in_subject_cannot_add_headers(smtp):
    send_email_report("a@x.com", "Hello\r\nBcc: evil@x.com", "body")
    message = smtp.sent[0]
    assert message["Bcc"] is None
    assert "\n" not in message["Subject"]


def test_wrong_gmail_password_gives_helpful_message(smtp):
    smtp.login_error = smtplib.SMTPAuthenticationError(535, b"bad credentials")
    with pytest.raises(RuntimeError, match="App Password"):
        send_email_report("a@x.com", "Hi", "body")


def test_network_problem_gives_readable_message(smtp):
    smtp.login_error = OSError("network down")
    with pytest.raises(RuntimeError, match="Could not send"):
        send_email_report("a@x.com", "Hi", "body")


def test_invalid_recipient_is_rejected_before_connecting(smtp):
    with pytest.raises(ValueError):
        send_email_report("nope", "Hi", "body")
    assert smtp.sent == []


def test_email_is_configured_flag(monkeypatch):
    for name in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    assert email_is_configured() is False

    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_USERNAME", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    assert email_is_configured() is True
