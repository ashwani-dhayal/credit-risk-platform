"""Talk-to-data agent: turn natural-language questions into SQL + answer.

Pipeline:
  1. Build prompt with pinned schema + few-shot examples.
  2. Call configured LLM (OpenAI / Groq / Gemini).
  3. Parse JSON, extract `sql`.
  4. Validate + harden SQL (src.utils.sql_safety).
  5. Execute against read-only SQLite connection.
  6. Ask LLM to summarise the result rows in plain English.

Fallback (no API key set):
  A small regex-based intent parser handles 6 canned questions so the demo
  still works offline. Each intent maps to a curated SQL template — same
  guardrails apply.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data.loader import get_table_schema, open_connection
from src.llm.client import LLMUnavailable, active_provider, get_client
from src.llm.prompts import build_messages, build_summary_messages
from src.utils.sql_safety import validate_and_harden


@dataclass
class AgentAnswer:
    question: str
    sql: str
    rows: list[dict]
    answer: str
    provider: str          # which backend produced the SQL
    used_fallback: bool    # True if the deterministic intent parser was used
    error: Optional[str] = None


def _extract_json_sql(text: str) -> str:
    """Extract the SQL string from the model output, tolerating fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json|sql)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "sql" in obj:
            return str(obj["sql"]).strip()
    except json.JSONDecodeError:
        pass

    # Last-resort: find the first {...} block.
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "sql" in obj:
                return str(obj["sql"]).strip()
        except json.JSONDecodeError:
            pass

    # If the model just emitted raw SQL.
    if text.lower().lstrip().startswith(("select", "with")):
        return text
    raise ValueError(f"Could not extract SQL from LLM response: {text[:200]}")


# -------------------------- Fallback intent parser --------------------------
# IMPORTANT: order matters — more specific patterns MUST come before
# generic ones. The first match wins.
_FALLBACK_INTENTS: list[tuple[re.Pattern, str]] = [
    # ---- Specific groupings (must come before the generic "default rate") ----
    (
        re.compile(r"\b(by|per)\b.*\b(education|education level)\b", re.I),
        (
            "SELECT NAME_EDUCATION_TYPE, "
            "ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct, COUNT(*) AS n "
            "FROM applications GROUP BY NAME_EDUCATION_TYPE "
            "ORDER BY default_rate_pct DESC"
        ),
    ),
    (
        re.compile(r"\b(by|per|top)\b.*\b(occupations?|jobs?|professions?)\b", re.I),
        (
            "SELECT OCCUPATION_TYPE, "
            "ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct, COUNT(*) AS n "
            "FROM applications WHERE OCCUPATION_TYPE IS NOT NULL "
            "GROUP BY OCCUPATION_TYPE HAVING COUNT(*) >= 50 "
            "ORDER BY default_rate_pct DESC LIMIT 10"
        ),
    ),
    (
        re.compile(r"\b(by|per)\b.*\bgender\b", re.I),
        (
            "SELECT CODE_GENDER, "
            "ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct, COUNT(*) AS n "
            "FROM applications GROUP BY CODE_GENDER ORDER BY default_rate_pct DESC"
        ),
    ),
    (
        re.compile(r"\b(by|per)\b.*\b(housing|house)\b", re.I),
        (
            "SELECT NAME_HOUSING_TYPE, "
            "ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct, COUNT(*) AS n "
            "FROM applications GROUP BY NAME_HOUSING_TYPE ORDER BY default_rate_pct DESC"
        ),
    ),
    # ---- Income / financial questions ----
    (
        re.compile(r"\baverage\s+(income|amt_income)|defaulters?\s+vs\b", re.I),
        (
            "SELECT TARGET, "
            "ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income, COUNT(*) AS n "
            "FROM applications GROUP BY TARGET"
        ),
    ),
    (
        re.compile(r"(income|amt_income).*(distribution|range|histogram|spread)", re.I),
        (
            "SELECT "
            "ROUND(MIN(AMT_INCOME_TOTAL),0) AS min_income, "
            "ROUND(AVG(AMT_INCOME_TOTAL),0) AS avg_income, "
            "ROUND(MAX(AMT_INCOME_TOTAL),0) AS max_income "
            "FROM applications"
        ),
    ),
    (
        re.compile(r"(credit.*income|loan.*income).*ratio", re.I),
        (
            "SELECT TARGET, "
            "ROUND(AVG(AMT_CREDIT * 1.0 / AMT_INCOME_TOTAL), 2) AS avg_ratio, "
            "COUNT(*) AS n FROM applications WHERE AMT_INCOME_TOTAL > 0 "
            "GROUP BY TARGET"
        ),
    ),
    # ---- Generic counts and rates (MUST come last) ----
    (
        re.compile(r"how many\b.*\b(applicants|rows|records|clients|loans)", re.I),
        "SELECT COUNT(*) AS total_applicants FROM applications",
    ),
    (
        re.compile(r"(overall\s+)?default rate|share of defaults|fraction.*default", re.I),
        "SELECT ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct FROM applications",
    ),
]


def _fallback_sql(question: str) -> Optional[str]:
    for pat, sql in _FALLBACK_INTENTS:
        if pat.search(question):
            return sql
    return None


def _summarise(question: str, sql: str, df: pd.DataFrame, provider: str) -> str:
    """Use the LLM to produce a 2-4 sentence answer; fall back to a template."""
    rows = df.head(20).to_dict(orient="records")
    if provider == "fallback":
        return _format_table_summary(df)
    try:
        client = get_client()
        return client.chat(
            build_summary_messages(question, json.dumps(rows, default=str), sql)
        )
    except Exception:
        return _format_table_summary(df)


def _format_table_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows matched the question."
    if df.shape == (1, 1):
        col = df.columns[0]
        val = df.iloc[0, 0]
        return f"{col.replace('_', ' ').title()}: {val}"
    return f"Returned {len(df)} row(s). See the table for details."


# ----------------------------- Public API -----------------------------
def answer(question: str) -> AgentAnswer:
    schema = get_table_schema()
    provider = active_provider()
    used_fallback = provider == "fallback"
    sql_raw: str

    if used_fallback:
        sql_raw = _fallback_sql(question) or (
            "SELECT 'unanswerable_with_fallback_parser' AS note"
        )
    else:
        try:
            client = get_client()
            llm_text = client.chat(build_messages(schema, question))
            sql_raw = _extract_json_sql(llm_text)
        except (LLMUnavailable, Exception) as e:
            # Hard-fall-through to the deterministic parser.
            used_fallback = True
            provider = "fallback"
            sql_raw = _fallback_sql(question) or (
                "SELECT 'llm_failed_and_no_fallback_match' AS note"
            )

    validation = validate_and_harden(sql_raw)
    if not validation.ok:
        return AgentAnswer(
            question=question,
            sql=sql_raw,
            rows=[],
            answer=f"I rejected the generated SQL: {validation.reason}",
            provider=provider,
            used_fallback=used_fallback,
            error=validation.reason,
        )

    try:
        with open_connection() as conn:
            df = pd.read_sql_query(validation.sql, conn)
    except sqlite3.Error as e:
        return AgentAnswer(
            question=question,
            sql=validation.sql,
            rows=[],
            answer=f"SQL execution failed: {e}",
            provider=provider,
            used_fallback=used_fallback,
            error=str(e),
        )

    summary = _summarise(question, validation.sql, df, provider)
    return AgentAnswer(
        question=question,
        sql=validation.sql,
        rows=df.to_dict(orient="records"),
        answer=summary,
        provider=provider,
        used_fallback=used_fallback,
    )
