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


# General knowledge answers for common credit-related questions that
# don't need SQL. First match wins.
_KNOWLEDGE_BASE = [
    (
        re.compile(r"what\s+(is|are)\s+credit\s*risk", re.I),
        (
            "**Credit risk** is the probability that a borrower will fail to repay a loan or "
            "meet their financial obligations, resulting in a loss for the lender.\n\n"
            "**Key aspects of credit risk:**\n"
            "- **Default risk** — the borrower stops paying entirely\n"
            "- **Concentration risk** — too many loans in one segment\n"
            "- **Downgrade risk** — the borrower's creditworthiness deteriorates over time\n\n"
            "**How banks manage it:**\n"
            "1. Credit scoring models (like the one in this platform) predict default probability\n"
            "2. Risk-based pricing — riskier borrowers pay higher interest rates\n"
            "3. Collateral requirements for high-risk applicants\n"
            "4. Portfolio diversification across industries, geographies, and loan sizes\n\n"
            "In this platform, we quantify credit risk as the **probability of default** — "
            "a number between 0% and 100% predicted by our LightGBM model based on the "
            "applicant's demographics, financials, and external credit bureau scores."
        ),
    ),
    (
        re.compile(r"what\s+(is|are)\s+(a\s+)?credit\s*score", re.I),
        (
            "A **credit score** is a numerical representation (typically 300-900 in India, "
            "300-850 in the US) of how likely you are to repay debt on time.\n\n"
            "**What goes into it:**\n"
            "- **Payment history (35%)** — do you pay EMIs/bills on time?\n"
            "- **Credit utilization (30%)** — how much of your available credit are you using?\n"
            "- **Credit age (15%)** — how long have you had credit accounts?\n"
            "- **Credit mix (10%)** — variety of loan types (home, car, card)\n"
            "- **New inquiries (10%)** — how often you've applied for new credit recently\n\n"
            "**Score ranges (CIBIL India):** 750+ is excellent, 650-749 is good, "
            "550-649 is fair, below 550 needs improvement.\n\n"
            "In our model, the `EXT_SOURCE_1/2/3` columns represent normalized external "
            "credit bureau scores — they're the single strongest predictors of default."
        ),
    ),
    (
        re.compile(r"what\s+(is|does)\s+(this|the)\s+(platform|app|tool|system)", re.I),
        (
            "This is an **AI-Powered Credit Risk Intelligence Platform** that helps "
            "lenders make better loan decisions. Here's what it does:\n\n"
            "1. **EDA** — Visualizes patterns in historical loan data (default rates by "
            "demographics, income levels, education, etc.)\n"
            "2. **Risk Prediction** — Scores new applicants using a LightGBM ML model "
            "(outputs probability of default + Approve/Review/Reject)\n"
            "3. **Explainability** — Shows SHAP values explaining exactly WHY each "
            "prediction was made\n"
            "4. **Decision Rules** — Converts complex ML patterns into human-readable "
            "IF-THEN business rules\n"
            "5. **Talk-to-Data** — This chatbot! Ask plain-English questions about the "
            "loan portfolio and get SQL-backed answers\n"
            "6. **Score Improvement** — Tips on how applicants can improve their credit profile\n\n"
            "Built with: LightGBM, SHAP, Streamlit, SQLite, and LLM-powered NL-to-SQL."
        ),
    ),
    (
        re.compile(r"what\s+(is|are)\s+(a\s+)?default\s*(rate)?", re.I),
        (
            "**Default** means a borrower has failed to repay their loan as agreed. The "
            "**default rate** is the percentage of borrowers who defaulted out of all borrowers.\n\n"
            "In our dataset:\n"
            "- TARGET = 1 means the applicant **defaulted**\n"
            "- TARGET = 0 means they **repaid on time**\n"
            "- The overall default rate is approximately **8%** (meaning 92% of people repay successfully)\n\n"
            "An 8% default rate is typical for consumer lending — it means for every 100 loans "
            "issued, about 8 will eventually go bad. The bank needs to price its interest rates "
            "to cover these expected losses while still making a profit."
        ),
    ),
    (
        re.compile(r"how\s+does\s+(the\s+)?model\s+work|how\s+(do|does)\s+(you|it)\s+predict", re.I),
        (
            "**How the prediction model works:**\n\n"
            "1. **Data input** — The applicant's details (income, loan amount, age, education, "
            "employment, credit scores, etc.) are collected\n\n"
            "2. **Feature engineering** — Raw data is transformed into meaningful ratios:\n"
            "   - Credit/Income ratio (loan amount ÷ annual income)\n"
            "   - Annuity/Income ratio (yearly payment ÷ income)\n"
            "   - Employment tenure in years\n"
            "   - Mean external credit score\n\n"
            "3. **LightGBM model** — A gradient-boosted decision tree ensemble trained on "
            "10,000+ historical applications learns which patterns lead to default\n\n"
            "4. **Output** — A probability between 0-100%, then mapped to:\n"
            "   - **< 20%** → Low risk → Approve\n"
            "   - **20-50%** → Medium risk → Manual Review\n"
            "   - **> 50%** → High risk → Reject\n\n"
            "5. **Explanation** — SHAP values show which features pushed the score up/down\n\n"
            "The model achieves **ROC-AUC of 0.895** and **KS statistic of 0.657** on validation data."
        ),
    ),
    (
        re.compile(r"what\s+(is|are)\s+shap|explain.*shap|what.*explainab", re.I),
        (
            "**SHAP (SHapley Additive exPlanations)** is a method from game theory that "
            "explains individual predictions by assigning each feature a contribution score.\n\n"
            "**How it works in this platform:**\n"
            "- For each prediction, SHAP calculates how much each feature pushed the risk "
            "score up or down from the average\n"
            "- Positive SHAP value = feature **increases** default risk\n"
            "- Negative SHAP value = feature **decreases** default risk\n\n"
            "**Example:** If an applicant has a high credit-to-income ratio, SHAP might "
            "assign it +0.15 (increases risk). If they have a high external credit score, "
            "SHAP assigns it -0.30 (decreases risk).\n\n"
            "**Why it matters:** Regulators and compliance teams require that banks can "
            "explain WHY a loan was rejected. SHAP provides that legally-defensible explanation."
        ),
    ),
]


def _check_knowledge_base(question):
    """If the question is general knowledge, return a direct answer."""
    for pat, response in _KNOWLEDGE_BASE:
        if pat.search(question):
            return response
    return None


# Patterns that indicate a question is NOT about querying the dataset.
# These should go to the general LLM, not the SQL pipeline.
_NON_DATA_PATTERNS = [
    re.compile(r"^(hi|hello|hey|howdy|greetings)\b", re.I),
    re.compile(r"^how\s+are\s+you", re.I),
    re.compile(r"^(what|whats)\s+(is\s+)?(the\s+)?(today|time|date|day)", re.I),
    re.compile(r"^(who|what)\s+are\s+you", re.I),
    re.compile(r"^(thank|thanks|bye|goodbye)", re.I),
    re.compile(r"^what\s+(is|are)\s+(a\s+|an\s+)?(loan|emi|interest|mortgage|collateral|npa|cibil|fico|banking|bank|finance|debt|equity|asset|liability|budget|savings?|investment|mutual fund|stock|bond|inflation|gdp|rbi|sebi|credit card|debit card|insurance|premium|risk|portfolio|diversif|amortiz|securiti|liquidity|solvency|capital|revenue|profit|loss|balance sheet|cash flow|roi|roe|eps|pe ratio|dividend|compound interest|simple interest|fixed deposit|recurring deposit|net worth|ipo|bull market|bear market|recession|depression|fiscal|monetary|tax|gst|income tax|tds)\b", re.I),
    re.compile(r"^(explain|define|describe|tell me about|what do you mean by)\s+", re.I),
    re.compile(r"^(how|why|when|where)\s+(do|does|did|can|could|should|would|is|are|was|were)\s+.{3,}(?!.*\b(applicants?|clients?|loans?|default|data|dataset|table|rows?|records?|count|average|total|sum|max|min|group|rate|percentage)\b)", re.I),
    re.compile(r"^(can you|could you|please)\s+(explain|tell|help|describe)", re.I),
    re.compile(r"\b(meaning|definition|concept|theory|principle)\b.*\??\s*$", re.I),
]


def _is_general_question(question):
    """Detect if a question is conversational/general rather than a data query."""
    q = question.strip()
    for pat in _NON_DATA_PATTERNS:
        if pat.search(q):
            return True
    return False


def _ask_llm_general(question):
    """Ask the LLM a general question directly (not SQL-related)."""
    try:
        client = get_client()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful credit risk and finance expert assistant. "
                    "Answer the user's question in a detailed, well-structured way. "
                    "Use markdown formatting (bold, bullet points, numbered lists) "
                    "to make your answer easy to read. Focus on credit risk, banking, "
                    "finance, and lending topics. If the question is completely unrelated "
                    "to finance or credit, still answer helpfully but briefly mention "
                    "that this platform specialises in credit risk analysis."
                ),
            },
            {"role": "user", "content": question},
        ]
        return client.chat(messages)
    except Exception:
        return None


def answer(question):
    # First check if this is a general knowledge question from our KB
    kb_answer = _check_knowledge_base(question)
    if kb_answer:
        return AgentAnswer(
            question=question,
            sql="-- No SQL needed (general knowledge question)",
            rows=[],
            answer=kb_answer,
            provider="knowledge_base",
            used_fallback=False,
        )

    # If it looks like a conversational/general question, route to Groq
    # directly instead of trying to make SQL out of it.
    if _is_general_question(question):
        general_answer = _ask_llm_general(question)
        if general_answer:
            return AgentAnswer(
                question=question,
                sql="-- General question (answered by AI assistant)",
                rows=[],
                answer=general_answer,
                provider=active_provider(),
                used_fallback=False,
            )

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
            used_fallback = True
            provider = "fallback"
            sql_raw = _fallback_sql(question) or (
                "SELECT 'llm_failed_and_no_fallback_match' AS note"
            )

    # If the LLM says unanswerable or the fallback didn't match, ask it as a
    # general knowledge question using the LLM directly
    if "unanswerable" in sql_raw.lower() or "no_fallback_match" in sql_raw.lower():
        general_answer = _ask_llm_general(question)
        if general_answer:
            return AgentAnswer(
                question=question,
                sql="-- General question (no SQL needed)",
                rows=[],
                answer=general_answer,
                provider=provider,
                used_fallback=False,
            )
        return AgentAnswer(
            question=question,
            sql=sql_raw,
            rows=[],
            answer="I couldn't find a relevant answer for this question in the dataset or my knowledge.",
            provider=provider,
            used_fallback=used_fallback,
            error="unanswerable",
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
