"""SQL safety guard for the talk-to-data agent.

The LLM prompt asks for a single SELECT, but we don't trust it. This
module enforces the same rules statically before anything hits SQLite.
"""

import re
from dataclasses import dataclass

import sqlparse

ALLOWED_TABLES = {"applications"}

FORBIDDEN_PATTERNS = [
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke)\b",
    r"\battach\b",
    r"\bdetach\b",
    r"\bpragma\b",
    r"\bvacuum\b",
    r"\bload_extension\b",
    r"\b\.shell\b",
    r"\b\.system\b",
]
MAX_ROWS = 200


@dataclass
class SqlValidation:
    ok: bool
    sql: str
    reason: str = ""


def validate_and_harden(sql):
    raw = sql.strip().rstrip(";").strip()
    if not raw:
        return SqlValidation(False, "", "Empty SQL.")

    if ";" in raw:
        return SqlValidation(False, raw, "Multiple statements are not allowed.")

    lowered = raw.lower()
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, lowered):
            return SqlValidation(False, raw, f"Forbidden keyword detected ({pat}).")

    parsed = sqlparse.parse(raw)
    if len(parsed) != 1:
        return SqlValidation(False, raw, "Exactly one statement is required.")

    first_token = parsed[0].token_first(skip_ws=True, skip_cm=True)
    if first_token is None:
        return SqlValidation(False, raw, "Could not parse SQL.")

    head = first_token.normalized.upper()
    if head not in {"SELECT", "WITH"}:
        return SqlValidation(False, raw, "Only SELECT/WITH queries are permitted.")

    # Recognise CTE aliases so `WITH t AS (...) SELECT * FROM t` works.
    cte_aliases = set(
        re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", lowered)
    )
    table_refs = set(re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered))
    table_refs |= set(re.findall(r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered))
    unknown = table_refs - ALLOWED_TABLES - cte_aliases
    if unknown:
        return SqlValidation(
            False, raw, f"Reference to disallowed table(s): {sorted(unknown)}"
        )

    # If the query has no LIMIT, add one so a runaway aggregate can't
    # return 300k rows to the UI.
    hardened = raw
    if not re.search(r"\blimit\s+\d+\b", lowered):
        hardened = f"{raw}\nLIMIT {MAX_ROWS}"

    return SqlValidation(True, hardened)
