import re

# Statement keywords that must never appear in a read-only query.
# (They are checked AFTER string literals, quoted names and comments are
#  blanked out, so 'set' inside text or a "Load" column is not a problem.)
BLOCKED_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|COPY|EXPORT|"
    r"IMPORT|ATTACH|DETACH|INSTALL|LOAD|PRAGMA|CALL|SET|RESET|VACUUM|"
    r"INTO|USE|CHECKPOINT"
    r")\b",
    re.IGNORECASE,
)

# Functions that read files / run other SQL / look at the system.
# Only blocked when used as a function call, so a column called
# "read_count" is still fine.
BLOCKED_FUNCTIONS = re.compile(
    r"\b("
    r"read_\w+|\w+_scan|glob|sniff_csv|duckdb_\w+|pragma_\w+|parquet_\w+|"
    r"query|query_table|getenv|current_setting|list_files?"
    r")\s*\(",
    re.IGNORECASE,
)

# FROM 'some_file.csv'  (DuckDB reads the file directly)
FILE_IN_FROM = re.compile(r"\b(FROM|JOIN)\s*'", re.IGNORECASE)


def clean_sql(sql: str) -> str:
    """Remove markdown formatting from an AI-generated SQL query."""
    query = sql.strip()

    if query.startswith("```"):
        query = re.sub(r"^```(?:sql)?", "", query, flags=re.IGNORECASE).strip()
        query = re.sub(r"```$", "", query).strip()

    return query


def _mask(sql: str) -> str:
    """Return the SQL with comments removed and quoted text blanked out.

    'it''s;here'  ->  ''
    "My Col"      ->  ""
    -- comment    ->  (space)
    """
    out = []
    i, n = 0, len(sql)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if ch == "-" and nxt == "-":
            end = sql.find("\n", i)
            i = n if end == -1 else end
            out.append(" ")

        elif ch == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")

        elif ch in ("'", '"'):
            quote = ch
            out.append(quote)
            i += 1

            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:  # escaped quote
                        i += 2
                        continue
                    break
                i += 1

            out.append(quote)
            i += 1

        else:
            out.append(ch)
            i += 1

    return "".join(out)


def validate_read_only_sql(sql: str) -> tuple[bool, str]:
    """Allow one read-only SELECT/WITH query and reject unsafe SQL.

    This is the first layer of defence. The second (stronger) layer is in
    db_manager.build_database(), which switches off DuckDB file access.
    """
    query = clean_sql(sql)

    if not query:
        return False, "AI did not generate any SQL."

    if "$$" in query:
        return False, "Dollar-quoted strings are not allowed."

    masked = _mask(query).strip()

    if masked.endswith(";"):
        masked = masked[:-1].rstrip()

    if ";" in masked:
        return False, "Only one SQL query is allowed."

    if not re.match(r"^(SELECT|WITH)\b", masked, flags=re.IGNORECASE):
        return False, "Only SELECT queries are allowed."

    keyword = BLOCKED_KEYWORDS.search(masked)
    if keyword:
        return False, f"This query contains a blocked SQL keyword: {keyword.group(1).upper()}."

    if BLOCKED_FUNCTIONS.search(masked) or FILE_IN_FROM.search(masked):
        return False, "File access and system functions are not allowed."

    cleaned = query.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()

    return True, cleaned
