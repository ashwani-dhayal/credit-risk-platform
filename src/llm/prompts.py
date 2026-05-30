"""Prompts used by the talk-to-data agent."""

SYSTEM_PROMPT = """You are a SQL analyst for a credit-risk platform.

You translate one natural-language question into ONE SQLite SELECT query
against the table `applications` and nothing else.

Hard rules:
- Output a JSON object only: {"sql": "<query>"}. No prose, no code fences.
- Use ONLY the table `applications` and the columns listed in the schema.
- The query MUST be a single SELECT (or WITH ... SELECT).
- Never use INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA.
- Always alias aggregates (e.g. AVG(...) AS avg_income).
- Cap to LIMIT 50 unless the question explicitly asks for more.
- Use proper SQLite syntax (e.g. CAST(... AS REAL), no PostgreSQL extensions).
- TARGET = 1 means defaulted, TARGET = 0 means repaid.
- DAYS_BIRTH and DAYS_EMPLOYED are negative integers (days before application).
  Compute age in years as (-DAYS_BIRTH / 365.25).

If the question cannot be answered with the schema, return
{"sql": "SELECT 'unanswerable_with_schema' AS note"}."""

# Few-shot examples. Kept short on purpose so input tokens stay low.
FEW_SHOT = [
    {
        "q": "How many applicants are there in total?",
        "sql": "SELECT COUNT(*) AS total_applicants FROM applications LIMIT 50",
    },
    {
        "q": "What is the overall default rate?",
        "sql": (
            "SELECT ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct "
            "FROM applications LIMIT 50"
        ),
    },
    {
        "q": "Average income by education level for clients who defaulted",
        "sql": (
            "SELECT NAME_EDUCATION_TYPE, "
            "ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income, "
            "COUNT(*) AS n "
            "FROM applications WHERE TARGET = 1 "
            "GROUP BY NAME_EDUCATION_TYPE "
            "ORDER BY avg_income DESC LIMIT 50"
        ),
    },
    {
        "q": "Show the 5 occupations with the highest default rate (min 100 clients)",
        "sql": (
            "SELECT OCCUPATION_TYPE, "
            "ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct, "
            "COUNT(*) AS n "
            "FROM applications WHERE OCCUPATION_TYPE IS NOT NULL "
            "GROUP BY OCCUPATION_TYPE HAVING COUNT(*) >= 100 "
            "ORDER BY default_rate_pct DESC LIMIT 5"
        ),
    },
]


def build_messages(schema_text, question):
    examples = "\n".join(
        f'Q: {ex["q"]}\nA: {{"sql": "{ex["sql"]}"}}' for ex in FEW_SHOT
    )
    user = (
        f"SCHEMA:\n{schema_text}\n\n"
        f"EXAMPLES:\n{examples}\n\n"
        f"Question: {question}\n"
        'Respond with ONLY: {"sql": "..."}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


SUMMARISE_SYSTEM = """You are a concise business analyst. Given a user
question and the SQL result rows (compact JSON), reply in 2-4 sentences with
the answer in plain English. State concrete numbers from the rows. Do NOT
invent numbers that are not in the rows. If the result is empty, say so."""


def build_summary_messages(question, rows_json, sql):
    return [
        {"role": "system", "content": SUMMARISE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"SQL: {sql}\n"
                f"Rows: {rows_json}\n"
                "Answer:"
            ),
        },
    ]
