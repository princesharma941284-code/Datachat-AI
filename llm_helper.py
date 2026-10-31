import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def has_api_key() -> bool:
    """Check whether an OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


_client = None


def get_client() -> OpenAI:
    """Create an OpenAI client once and reuse it."""
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to the .env file."
            )

        _client = OpenAI(api_key=api_key)

    return _client


try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    _TRANSIENT_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except ImportError:
    _TRANSIENT_ERRORS = ()

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _is_transient(error: Exception) -> bool:
    """True for temporary problems worth retrying (busy, rate limit, network)."""
    if isinstance(error, _TRANSIENT_ERRORS):
        return True

    return getattr(error, "status_code", None) in _TRANSIENT_STATUS_CODES


def get_model_name() -> str:
    """Return the OpenAI model configured in .env."""
    return os.getenv("OPENAI_MODEL", "gpt-5.6-luna")


def _generate_with_retry(
    instructions: str,
    user_input: str,
    attempts: int = 4,
) -> str:
    """Call OpenAI and retry temporary API failures."""

    for attempt in range(attempts):
        try:
            response = get_client().responses.create(
                model=get_model_name(),
                instructions=instructions,
                input=user_input,
            )

            return response.output_text.strip()

        except Exception as error:
            # Wrong key / wrong model / bad request: fail at once.
            if not _is_transient(error) or attempt == attempts - 1:
                raise

            time.sleep(2 * (attempt + 1))


def generate_sql(
    question: str,
    schema: str,
    history: str = "",
) -> str:
    """Convert a user's question into one safe DuckDB SQL query."""

    instructions = """
You are DataChat AI, a careful SQL analyst.

Rules:
- Use only the tables and columns shown in the provided schema.
- If the question needs data from more than one table, use a JOIN.
- If the schema lists Relationships, use exactly those columns in JOIN conditions.
- If the schema lists Business metrics and the question uses a metric name or
  one of its synonyms (for example "sales" for revenue), use that exact formula.
- For a category column with listed values, copy the value exactly as listed
  (same spelling and capital letters) when you filter on it.
- Generate exactly one valid DuckDB read-only SELECT query.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, or PRAGMA.
- Never use file or system functions such as read_csv, read_text, glob.
- Put double quotes around table or column names that contain spaces or
  special characters, for example "Order Date".
- Date columns have DATE or TIMESTAMP type. If a date column is VARCHAR,
  CAST it first. Use date_trunc or strftime to group by month or year.
- For text matching use ILIKE so capital letters do not matter.
- Add LIMIT 1000 unless the query is an aggregate or the user asks for all rows.
- Use previous questions only to understand follow-up questions.
- Return only SQL.
- Do not use markdown, explanations, or code fences.
- Format SQL clearly on multiple lines.
"""

    history_block = (
        f"\nPrevious questions in this chat:\n{history}\n"
        if history
        else ""
    )

    user_input = f"""
Database schema:
{schema}
{history_block}
User question:
{question}
"""

    return _generate_with_retry(instructions, user_input)


def explain_results(
    question: str,
    result: pd.DataFrame,
) -> str:
    """Explain the SQL result using only the displayed result data."""

    if result.empty:
        return "No matching data was found for this question."

    result_text = result.head(30).to_csv(index=False)

    instructions = """
You are a business data analyst.

Explain the query result in simple language.

Rules:
- Use only the result data given to you.
- Do not invent numbers, causes, or facts.
- Keep the answer short.
- Maximum three bullet points.
"""

    user_input = f"""
Question:
{question}

Query result:
{result_text}
"""

    return _generate_with_retry(instructions, user_input)

