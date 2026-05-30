"""SQL safety guardrails for the talk-to-data chatbot.

The LLM is constrained by the prompt to emit a single read-only SELECT,
but we *also* enforce that statically before execution:

1. Single-statement only (no `;` chaining).
2. Must start with SELECT or WITH (CTE).
3. Forbidden keywords are blocked (DDL, DML, attach, pragma, etc.).
4. Only the allowed table can be referenced.
5. Hard LIMIT cap injected if missing.
"""

from __future__ import annotations

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


def validate_and_harden(sql: str) -> SqlValidation:
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

    # Collect CTE aliases (so `WITH t AS (... applications ...)` works).
    cte_aliases = set(
        re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", lowered)
    )
    # Reject any reference to unknown tables (cheap heuristic).
    table_refs = set(re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered))
    table_refs |= set(re.findall(r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered))
    unknown = table_refs - ALLOWED_TABLES - cte_aliases
    if unknown:
        return SqlValidation(
            False, raw, f"Reference to disallowed table(s): {sorted(unknown)}"
        )

    # Inject a LIMIT cap if the query doesn't have one.
    hardened = raw
    if not re.search(r"\blimit\s+\d+\b", lowered):
        hardened = f"{raw}\nLIMIT {MAX_ROWS}"

    return SqlValidation(True, hardened)
