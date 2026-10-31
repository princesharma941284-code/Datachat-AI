import pandas as pd
import pytest

from day3_features import detect_anomalies, make_forecast


def test_detect_anomalies_flags_extreme_value():
    df = pd.DataFrame({"sales": [10, 11, 9, 10, 100]})
    result = detect_anomalies(df, "sales", threshold=2.0)
    assert result["_is_anomaly"].sum() == 1
    assert result.iloc[-1]["_is_anomaly"]


def test_forecast_returns_requested_periods():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "sales": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        }
    )
    result = make_forecast(df, "date", "sales", periods=3)
    assert len(result) == 3
    assert result["sales_forecast"].iloc[0] > 28


def test_forecast_rejects_too_few_rows():
    df = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=2), "sales": [1, 2]}
    )
    with pytest.raises(ValueError):
        make_forecast(df, "date", "sales")
