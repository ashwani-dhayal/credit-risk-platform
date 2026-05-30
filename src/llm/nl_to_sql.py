"""Natural-language question -> safe SQL -> rows -> plain-English answer.

The pipeline: prompt the LLM, extract JSON, validate the SQL with
sql_safety, run it read-only against SQLite, then ask the LLM to
summarise the rows. If no LLM key is configured (or if anything along
the way breaks) we fall back to a regex-based intent parser that
covers nine canonical questions.
"""

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
    rows: list
    answer: str
    provider: str
    used_fallback: bool
    error: Optional[str] = None


def _extract_json_sql(text):
    """Pull a SQL string out of the model's response, tolerating fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json|sql)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "sql" in obj:
            return str(obj["sql"]).strip()
    except json.JSONDecodeError:
        pass

    # Try to find the first {...} block.
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "sql" in obj:
                return str(obj["sql"]).strip()
        except json.JSONDecodeError:
            pass

    # Last resort: maybe the model just emitted raw SQL.
    if text.lower().lstrip().startswith(("select", "with")):
        return text

    raise ValueError(f"Could not extract SQL from LLM response: {text[:200]}")


# ---- Fallback intent parser ----
# Specific patterns must come BEFORE the generic ones. First match wins.
_FALLBACK_INTENTS = [
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
            "SELECT "
            "CASE TARGET WHEN 1 THEN 'Defaulted' ELSE 'Repaid' END AS status, "
            "ROUND(AVG(AMT_CREDIT * 1.0 / AMT_INCOME_TOTAL), 2) AS avg_credit_income_ratio, "
            "COUNT(*) AS n "
            "FROM applications WHERE AMT_INCOME_TOTAL > 0 "
            "GROUP BY TARGET ORDER BY TARGET DESC"
        ),
    ),
    (
        re.compile(r"how many\b.*\b(applicants|rows|records|clients|loans)", re.I),
        "SELECT COUNT(*) AS total_applicants FROM applications",
    ),
    (
        re.compile(r"(overall\s+)?default rate|share of defaults|fraction.*default", re.I),
        "SELECT ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct FROM applications",
    ),
]


def _fallback_sql(question):
    for pat, sql in _FALLBACK_INTENTS:
        if pat.search(question):
            return sql
    return None


def _summarise(question, sql, df, provider):
    """Get a 2-4 sentence summary. Falls back to a template if the LLM fails."""
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


def _format_table_summary(df):
    if df.empty:
        return "No rows matched the question."
    if df.shape == (1, 1):
        col = df.columns[0]
        val = df.iloc[0, 0]
        return f"{col.replace('_', ' ').title()}: {val}"
    return f"Returned {len(df)} row(s). See the table for details."


def answer(question):
    schema = get_table_schema()
    provider = active_provider()
    used_fallback = provider == "fallback"

    if used_fallback:
        sql_raw = _fallback_sql(question) or (
            "SELECT 'unanswerable_with_fallback_parser' AS note"
        )
    else:
        try:
            client = get_client()
            llm_text = client.chat(build_messages(schema, question))
            sql_raw = _extract_json_sql(llm_text)
        except (LLMUnavailable, Exception):
            # LLM died -- use the deterministic parser instead of giving up.
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
