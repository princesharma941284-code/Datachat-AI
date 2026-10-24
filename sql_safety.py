import re

BLOCKED_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"MERGE|COPY|EXPORT|IMPORT|ATTACH|DETACH|INSTALL|LOAD|"
    r"PRAGMA|CALL|SET|RESET|VACUUM|INTO"
    r")\b",
    re.IGNORECASE,
)


def clean_sql(sql: str) -> str:
    """Remove markdown formatting from an AI-generated SQL query."""
    query = sql.strip()

    if query.startswith("```"):
        query = re.sub(r"^```(?:sql)?", "", query, flags=re.IGNORECASE).strip()
        query = re.sub(r"```$", "", query).strip()

    return query


def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    """Allow one read-only SELECT/WITH query and reject unsafe SQL."""
    query = clean_sql(sql)

    if not query:
        return False, "AI did not generate any SQL."

    if query.count(";") > 1 or (";" in query and not query.endswith(";")):
        return False, "Only one SQL query is allowed."

    query = query.rstrip(";").strip()

    if not re.match(r"^(SELECT|WITH)\b", query, flags=re.IGNORECASE):
        return False, "Only SELECT queries are allowed."

    if BLOCKED_KEYWORDS.search(query):
        return False, "This query contains a blocked SQL command."

    return True, query
