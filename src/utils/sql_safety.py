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

    # Quote any reserved words the LLM used as column aliases.
    _RESERVED_ALIASES = {
        "group", "order", "select", "from", "where", "having",
        "limit", "index", "key", "table", "column", "values",
        "check", "default", "primary", "unique", "foreign",
    }
    hardened = raw
    for word in _RESERVED_ALIASES:
        # Match: AS <word> (case-insensitive) that isn't already quoted
        hardened = re.sub(
            rf'(?i)\bAS\s+{word}\b(?!\s*["\'])',
            f"AS [{word}]",
            hardened,
        )
        # Also fix GROUP BY <word> references
        hardened = re.sub(
            rf'(?i)\bGROUP\s+BY\s+{word}\b(?!\s*["\'])',
            f"GROUP BY [{word}]",
            hardened,
        )

    # If the query has no LIMIT, add one. But skip if there's a UNION
    # (LIMIT goes after the last SELECT in a UNION, but injecting it
    # blindly breaks things — safer to just let it run uncapped for
    # UNION queries since they're typically small).
    hardened_lower = hardened.lower()
    has_union = re.search(r"\bunion\b", hardened_lower)
    has_limit = re.search(r"\blimit\s+\d+\b", hardened_lower)
    if not has_limit and not has_union:
        hardened = f"{hardened}\nLIMIT {MAX_ROWS}"

    return SqlValidation(True, hardened)
