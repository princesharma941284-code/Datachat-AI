"""Day 3 analytics: anomaly detection, simple forecasting, and email reports.

Uses only the existing project dependencies plus Python's standard library.
"""
from __future__ import annotations

import io
import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

import pandas as pd


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns that contain numeric data."""
    return dataframe.select_dtypes(include="number").columns.tolist()


def date_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns that are already datetime-like or can mostly be parsed."""
    columns: list[str] = []
    for column in dataframe.columns:
        series = dataframe[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            columns.append(column)
            continue
        if not pd.api.types.is_string_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        non_null = series.notna().sum()
        if non_null and parsed.notna().sum() >= 0.8 * non_null:
            columns.append(column)
    return columns


def detect_anomalies(
    dataframe: pd.DataFrame,
    value_column: str,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag unusually high/low numeric values using a z-score."""
    if value_column not in dataframe.columns:
        raise ValueError("Selected numeric column was not found.")

    values = pd.to_numeric(dataframe[value_column], errors="coerce")

    result = dataframe.copy()
    result["_anomaly_score"] = 0.0

    # Modified z-score is more robust than a normal z-score when one extreme
    # value is itself the anomaly and would otherwise inflate the standard
    # deviation.
    median = values.median()
    mad = (values - median).abs().median()

    if pd.isna(mad) or mad == 0:
        mean = values.mean()
        std = values.std()
        if pd.isna(std) or std == 0:
            result["_is_anomaly"] = False
            return result
        score = ((values - mean) / std).abs()
    else:
        score = (0.6745 * (values - median) / mad).abs()

    result["_anomaly_score"] = score
    result["_is_anomaly"] = result["_anomaly_score"] >= float(threshold)

    return result


def make_forecast(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = 7,
) -> pd.DataFrame:
    """Create a simple linear-trend forecast without adding a heavy ML package."""
    if periods < 1 or periods > 365:
        raise ValueError("Forecast periods must be between 1 and 365.")

    data = dataframe[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna().sort_values(date_column)

    if len(data) < 3:
        raise ValueError("At least 3 valid date/value rows are required.")

    # Aggregate duplicate dates so the trend is based on one value per date.
    data = data.groupby(date_column, as_index=False)[value_column].mean()

    if len(data) < 3:
        raise ValueError("At least 3 unique dates are required.")

    x = (data[date_column] - data[date_column].min()).dt.total_seconds() / 86400.0
    y = data[value_column].astype(float)

    x_mean = x.mean()
    y_mean = y.mean()
    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        slope = 0.0
    else:
        slope = ((x - x_mean) * (y - y_mean)).sum() / denominator

    intercept = y_mean - slope * x_mean

    last_date = data[date_column].max()
    inferred = pd.infer_freq(data[date_column])
    if inferred:
        future_dates = pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=inferred,
        )[1:]
    else:
        deltas = data[date_column].sort_values().diff().dropna()
        step = deltas.median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(days=1)
        future_dates = [last_date + step * i for i in range(1, periods + 1)]

    future_x = (
        pd.Series(future_dates) - data[date_column].min()
    ).dt.total_seconds() / 86400.0

    forecast_values = intercept + slope * future_x

    return pd.DataFrame(
        {
            date_column: pd.to_datetime(future_dates),
            f"{value_column}_forecast": forecast_values.round(2),
        }
    )


def build_report(
    title: str,
    dataframe: pd.DataFrame,
    max_rows: int = 100,
) -> str:
    """Turn a result DataFrame into a compact plain-text report."""
    preview = dataframe.head(max_rows)
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"Rows in report: {len(preview)}"
        + (f" (showing first {max_rows})" if len(dataframe) > max_rows else "")
        + "\n\n"
        + preview.to_string(index=False)
    )


_EMAIL_PATTERN = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")
MAX_RECIPIENTS = 10


def parse_recipients(text) -> list[str]:
    """Turn 'a@x.com, b@y.com' into a clean list. Raises ValueError for bad input.

    Only plain addresses are accepted (no names, no line breaks), so nobody
    can sneak extra email headers into the message.
    """
    if isinstance(text, (list, tuple)):
        parts = [str(item) for item in text]
    else:
        parts = re.split(r"[,;\s]+", str(text or ""))

    recipients = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not _EMAIL_PATTERN.match(part):
            raise ValueError(f"'{part}' is not a valid email address.")
        if part.lower() not in [r.lower() for r in recipients]:
            recipients.append(part)

    if not recipients:
        raise ValueError("Enter at least one recipient email address.")
    if len(recipients) > MAX_RECIPIENTS:
        raise ValueError(f"You can send to at most {MAX_RECIPIENTS} addresses at once.")
    return recipients


def email_is_configured() -> bool:
    """True when the SMTP_* settings needed to send mail are present."""
    return all(os.getenv(name) for name in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD"))


def _deliver(message, host, port, username, password, use_tls) -> None:
    context = ssl.create_default_context()

    try:
        if use_tls:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(username, password)
                server.send_message(message)
        else:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
                server.login(username, password)
                server.send_message(message)

    except smtplib.SMTPAuthenticationError as error:
        raise RuntimeError(
            "The email server rejected the login. For Gmail use an App Password "
            "(16 letters, no spaces) in SMTP_PASSWORD, not your normal password, "
            "and make sure 2-Step Verification is on."
        ) from error
    except smtplib.SMTPRecipientsRefused as error:
        raise RuntimeError("The email server refused the recipient address(es).") from error
    except (OSError, smtplib.SMTPException) as error:
        raise RuntimeError(f"Could not send the email: {error}") from error


def send_email_report(
    recipient,
    subject: str,
    report_text: str,
    dataframe: Optional[pd.DataFrame] = None,
    attachment_name: str = "datachat_report.csv",
) -> None:
    """Send a report through SMTP (for example Gmail).

    recipient: one address, several separated by commas, or a list.

    Settings come from environment variables (.env):
      Required: SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD   (SMTP_PORT default 587)
      Optional: SMTP_USE_TLS=true/false, SMTP_FROM
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() not in {"0", "false", "no"}
    sender = os.getenv("SMTP_FROM") or username

    if not all([host, username, password, sender]):
        raise RuntimeError(
            "Email is not configured. Set SMTP_HOST, SMTP_PORT, "
            "SMTP_USERNAME, SMTP_PASSWORD and optionally SMTP_FROM in .env."
        )

    recipients = parse_recipients(recipient)

    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = re.sub(r"[\r\n]+", " ", subject or "DataChat AI Report").strip()
    message.set_content(report_text)

    if dataframe is not None:
        message.add_attachment(
            dataframe.to_csv(index=False).encode("utf-8"),
            maintype="text",
            subtype="csv",
            filename=re.sub(r"[^A-Za-z0-9_.-]+", "_", attachment_name) or "datachat_report.csv",
        )

    _deliver(message, host, port, username, password, use_tls)
