import pandas as pd
import pytest

from day3_features import (
    build_report,
    date_columns,
    detect_anomalies,
    make_forecast,
    numeric_columns,
    send_email_report,
)


def test_constant_values_have_no_anomaly():
    result = detect_anomalies(pd.DataFrame({"x": [5, 5, 5, 5]}), "x")
    assert not result["_is_anomaly"].any()


def test_unknown_column_raises():
    with pytest.raises(ValueError):
        detect_anomalies(pd.DataFrame({"x": [1, 2, 3]}), "missing")


def test_text_values_do_not_crash_anomaly_detection():
    result = detect_anomalies(pd.DataFrame({"x": ["1", "2", "oops", "2", "1"]}), "x")
    assert len(result) == 5


def test_forecast_rejects_bad_period_count():
    frame = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=5), "v": range(5)})
    with pytest.raises(ValueError):
        make_forecast(frame, "d", "v", periods=0)
    with pytest.raises(ValueError):
        make_forecast(frame, "d", "v", periods=366)


def test_forecast_needs_three_unique_dates():
    frame = pd.DataFrame({"d": ["2026-01-01"] * 4 + ["2026-01-02"], "v": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        make_forecast(frame, "d", "v")


def test_forecast_follows_a_straight_line():
    frame = pd.DataFrame(
        {"d": pd.date_range("2026-01-01", periods=6), "v": [10, 20, 30, 40, 50, 60]}
    )
    result = make_forecast(frame, "d", "v", periods=2)
    assert result["v_forecast"].tolist() == [70.0, 80.0]


def test_column_finders():
    frame = pd.DataFrame({"n": [1, 2], "d": ["2026-01-01", "2026-01-02"], "t": ["a", "b"]})
    assert numeric_columns(frame) == ["n"]
    assert date_columns(frame) == ["d"]


def test_report_has_title_and_row_limit_note():
    frame = pd.DataFrame({"a": range(5)})
    text = build_report("Sales", frame, max_rows=2)
    assert text.startswith("Sales\n=====")
    assert "showing first 2" in text


def test_email_without_settings_gives_clear_error(monkeypatch):
    for name in ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM"]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="not configured"):
        send_email_report("a@b.com", "subject", "body")
