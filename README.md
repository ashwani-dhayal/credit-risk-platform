# 💳 AI-Powered Credit Risk Intelligence Platform

A lightweight, end-to-end credit-risk platform built on the
[Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data).
It combines **machine learning, explainable AI, and a natural-language SQL
agent** behind a single Streamlit UI, all containerised with Docker so an
evaluator can run it with one command.

> Submission for the **NeoStats AI Engineer** assignment.

---

## ✨ Features

| # | Module | What it does |
|---|---|---|
| 1 | **EDA** | Demographics, financials, missingness, default-driver charts. |
| 2 | **Talk-to-Data Chatbot** | Plain-English ⟶ safe SQLite SELECT ⟶ business-readable summary. Works with OpenAI, Groq, or Gemini; falls back to a deterministic intent parser when no key is set. |
| 3 | **ML Layer** | LightGBM classifier, class-imbalance handled with `scale_pos_weight`, ROC-AUC ~0.90, KS ~0.66 on the bundled sample. Outputs probability + Low/Medium/High band + Approve/Review/Reject decision. |
| 4 | **Explainable AI** | Per-prediction SHAP TreeExplainer values + global driver chart. |
| 5 | **Decision Rules** | Depth-limited tree → 12 readable IF-THEN rules with support, default-rate, and lift. |
| 6 | **UI** | Single Streamlit app with 6 sections (Overview / EDA / Predict / Explain / Rules / Chatbot). |
| 7 | **Dockerised** | `docker compose up` boots a pre-trained, ready-to-serve container. |

---

## 🏗 Architecture

```
┌─────────────────────────── Streamlit UI ───────────────────────────┐
│  Overview │ EDA │ Predict │ Explain │ Rules │ Talk-to-Data Chatbot │
└────┬───────────────┬──────────┬──────────┬──────────┬──────────────┘
     │               │          │          │          │
     │               │          ▼          ▼          ▼
     │               │     ┌──────────────────────────────┐
     │               │     │  src/ml      src/rules       │
     │               │     │  • LightGBM  • DecisionTree  │
     │               │     │  • SHAP      • RuleExtractor │
     │               │     └────────────┬─────────────────┘
     │               │                  │
     │               ▼                  ▼
     │     ┌────────────────────────────────────────────┐
     │     │    src/data  (loader, preprocess, schema)  │
     │     │    sample_generator (offline-friendly)     │
     │     └────────────┬───────────────────────────────┘
     │                  ▼
     │          ┌───────────────┐
     │          │  SQLite DB    │  data/processed/credit_risk.db
     │          └───────┬───────┘
     │                  ▲
     │                  │
     ▼                  │
┌──────────────────────────────────────────────────┐
│  src/llm                                         │
│   • client.py    — OpenAI / Groq / Gemini auto   │
│   • nl_to_sql.py — agent + safety guardrails     │
│   • prompts.py   — token-optimised templates     │
│   • src/utils/sql_safety.py — read-only SELECT   │
└──────────────────────────────────────────────────┘
```

The **SQLite DB is the single source of truth** for both ML and the chatbot,
so every chatbot answer is consistent with the model's training data.

---

## 🚀 Quick start (Docker — recommended)

```bash
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

# (optional) drop the real Kaggle CSV here for full-fidelity results:
#   data/raw/application_train.csv
# If absent, a synthetic 10k-row sample is bundled and used automatically.

cp .env.example .env          # then add ONE LLM key (OpenAI / Groq / Gemini)
docker compose up --build
```

Open <http://localhost:8501>. The container pre-trains the model at build
time, so the UI is fully functional from the first request — no warm-up.

> No LLM key? It still works. The chatbot transparently falls back to a
> deterministic intent parser that handles 9 canonical questions about the
> data.

---

## 🧪 Local development (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/generate_sample.py    # only if you didn't drop the Kaggle CSV
python scripts/build_db.py
python scripts/train_model.py
python scripts/derive_rules.py

streamlit run app/streamlit_app.py
```

Run the test suite:

```bash
pytest -q
```

---

## 📂 Repository layout

```
credit-risk-platform/
├── app/
│   └── streamlit_app.py              # Multi-section UI
├── data/
│   ├── raw/                          # Drop Kaggle CSVs here (gitignored)
│   ├── sample/                       # 10k-row synthetic CSV (committed)
│   └── processed/                    # Auto-built SQLite DB
├── documents/
│   └── presentation.pdf              # Solution deck (PDF)
├── models/                           # Trained artifacts + metrics + rules
│   ├── lgbm_model.joblib
│   ├── metrics.json
│   ├── classification_report.txt
│   └── rules.json
├── scripts/                          # CLI entry points
│   ├── generate_sample.py
│   ├── build_db.py
│   ├── train_model.py
│   └── derive_rules.py
├── src/
│   ├── config.py                     # Env-driven config (.env-aware)
│   ├── data/                         # Schema + loader + preprocessing
│   │   ├── schema.py
│   │   ├── sample_generator.py
│   │   ├── loader.py
│   │   └── preprocess.py
│   ├── ml/                           # Train / predict / SHAP
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── explain.py
│   ├── rules/derive.py               # Decision-tree → readable rules
│   ├── llm/                          # Talk-to-data agent
│   │   ├── client.py                 # OpenAI / Groq / Gemini auto-detect
│   │   ├── prompts.py                # Token-optimised templates
│   │   └── nl_to_sql.py              # Agent + fallback parser
│   └── utils/sql_safety.py           # Read-only SELECT guardrails
├── tests/
│   ├── test_sql_safety.py
│   └── test_predict.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔐 Environment variables

See [`.env.example`](.env.example) for the full list. The minimum useful
config is **one** of:

```bash
OPENAI_API_KEY=sk-...
# or
GROQ_API_KEY=gsk_...
# or
GEMINI_API_KEY=...
```

Auto-detection priority: **OpenAI → Groq → Gemini → fallback parser**.

---

## 🎯 Model details

| Item | Choice | Why |
|---|---|---|
| Algorithm | **LightGBM** | Best-in-class on tabular credit data; handles mixed types after one-hot; trains in seconds. |
| Class imbalance | `scale_pos_weight = neg/pos` | Outperforms naive over-/under-sampling on this dataset and avoids leaking synthetic samples into validation. |
| Validation split | 80/20 stratified | Preserves the 8% default rate in both folds. |
| Primary metric | **ROC-AUC** | The Kaggle competition metric and the most common credit-risk benchmark. |
| Secondary metrics | KS, PR-AUC, F1 @ Youden-optimal threshold | KS is the standard banking metric; PR-AUC is robust to imbalance. |
| Operating threshold | Threshold that maximises Youden's J on the validation set | Translates probability to a decision in a principled way. |
| Risk bands | Low (<0.20) / Medium (<0.50) / High (>=0.50) | Tunable via `RISK_LOW_MAX` / `RISK_MEDIUM_MAX`. |

### Reference results on the bundled 10k sample

| Metric | Value |
|---|---|
| ROC-AUC | **0.895** |
| PR-AUC | **0.511** |
| KS statistic | **0.657** |
| F1 @ optimal threshold | 0.399 |
| Default rate | 8.0% |
| Train / val rows | 8 000 / 2 000 |

(These regenerate on every `docker compose up --build`; with the full Kaggle
dataset you should expect ROC-AUC in the 0.74–0.78 range, since the real data
has more noise than the calibrated synthetic sample.)

---

## 🤖 Talk-to-Data: prompt engineering & hallucination control

1. **Pinned schema** — the table schema (~25 columns with one-line
   descriptions) is part of the system prompt; no DB introspection
   round-trip per turn.
2. **JSON-only output** — the LLM is forced to emit `{"sql": "..."}`. No
   prose, no fences. We parse JSON deterministically; if parsing fails we
   fall through to the regex parser instead of guessing.
3. **Few-shot, low cost** — only 4 short examples cover the common shapes
   (count, average, group-by, top-N).
4. **Temperature 0** — repeatable, no creative SQL.
5. **Token cap** — 256-token responses; SQL never benefits from prose.
6. **Static SQL guardrails** (`src/utils/sql_safety.py`):
   - Single statement only (no `;` chaining)
   - Must start with `SELECT` or `WITH`
   - Forbidden keywords: `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA/...`
   - Tables limited to `applications` (CTE aliases recognised)
   - Hard `LIMIT 200` injected if missing
7. **Read-only DB connection** at execution time (SQLite `mode=ro`).
8. **Result-grounded summary** — the second LLM call is told to use ONLY
   the rows we just fetched; the prompt explicitly forbids invented numbers.
9. **Deterministic fallback** — when no key is set or the LLM fails, the
   agent transparently switches to a regex intent parser that covers the
   evaluation's "5 working queries" requirement (we ship **9**).

### Sample working queries

| # | Question | What it returns |
|---|---|---|
| 1 | How many applicants are there in total? | Single row count |
| 2 | What is the overall default rate? | % defaulted |
| 3 | Default rate by education level | One row per education level |
| 4 | Top occupations by default rate | Top 10 occupations (≥50 clients) |
| 5 | Average income for defaulters vs non-defaulters | 2 rows by TARGET |
| 6 | Default rate by gender | F vs M |
| 7 | Default rate by housing type | One row per housing type |
| 8 | Income distribution range | min / avg / max |
| 9 | Credit-to-income ratio for defaulters vs non | 2 rows by TARGET |

---

## 📜 Rule derivation

We fit a **depth-4 decision tree** with `min_samples_leaf = 2%` of the
population on the same engineered features as the LightGBM model, then walk
each leaf to emit one IF-THEN rule:

```
Rule 13: IF CREDIT_INCOME_RATIO > 5.19 AND ANNUITY_INCOME_RATIO > 0.46
         AND EXT_SOURCE_MEAN <= 0.47 AND NAME_EDUCATION_TYPE != 'Higher education'
         THEN risk = High  (support=6.9%, default_rate=88.3%, lift=11.04×, n=686)
```

Rules are ranked by `|leaf_default_rate − base_rate| × support`, so we
surface the rules that move the most population the most.

---

## 🧠 Five EDA insights (full charts in the UI)

1. **External credit scores dominate.** `EXT_SOURCE_*` are the strongest
   single predictors; high scores cut the default rate by ~5×.
2. **Credit-to-income ratio matters more than absolute income.** Defaulters
   show meaningfully higher loan-to-income ratios.
3. **Education is a strong demographic signal.** Higher-education and
   Academic-degree clients default at noticeably lower rates.
4. **Age skew.** Younger applicants (≤30y) carry higher default
   probability; the curve flattens after ~45.
5. **Employment stability protects.** Longer employment tenure lowers
   default rate; the unemployed sentinel `DAYS_EMPLOYED = 365243` MUST be
   treated as missing or it dominates the signal.

---

## 🧯 Known limitations & possible improvements

- **Sample data only by default**: ROC-AUC numbers come from a calibrated
  synthetic sample. Drop `application_train.csv` into `data/raw/` for full
  fidelity. The pipeline auto-detects and uses it.
- **Only `application_train` is modelled.** The Kaggle competition includes
  `bureau`, `previous_application`, etc. Adding a feature-aggregation pass
  on those tables would lift AUC by another ~0.03.
- **No model monitoring.** A production deployment would add data-drift
  monitoring (Evidently / WhyLogs) and periodic retraining.
- **Single-tenant SQLite.** Fine for a demo; switch to Postgres for
  multi-user concurrency.
- **No authentication on the UI.** Streamlit's auth is enterprise-tier;
  put it behind a reverse proxy in production.
- **LLM cost cap not enforced.** Token usage is bounded by `max_tokens`
  but per-user daily quotas are not. Add a Redis-backed rate limiter for
  production.

---

## 🧪 Major design decisions (1-line each)

- **Streamlit, not FastAPI + React** — single-process demo, fewer moving
  parts, evaluator-friendly.
- **SQLite as the data plane** — zero-config, file-based, supported by the
  same Python the model uses.
- **Multi-provider LLM client** — evaluator might have any of OpenAI / Groq
  / Gemini keys; we autodetect.
- **Deterministic fallback for the chatbot** — the demo NEVER fails to
  answer the canonical 5 questions, even offline.
- **SHAP TreeExplainer** — exact Shapley values for tree ensembles, no
  approximation.
- **Pre-train inside the Docker image** — no runtime "first-request is
  slow" surprise.

---

## 📑 License

This project is provided for the NeoStats AI Engineer assignment. No
warranty.
