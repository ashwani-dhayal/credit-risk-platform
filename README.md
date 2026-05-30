# Credit Risk Intelligence Platform

End-to-end credit-risk app for the **NeoStats AI Engineer** assignment.
It loads the Home Credit Default Risk data, trains a default-prediction
model, explains every prediction with SHAP, derives readable
IF-THEN rules, and lets you ask plain-English questions through a
talk-to-data chatbot. Everything lives behind one Streamlit UI and
ships in a Docker image you can boot with one command.

<p align="center">
  <img src="documents/screenshots/01_overview.png" alt="UI overview" width="80%" />
</p>


## Contents

1. [What it does](#what-it-does)
2. [Screenshots](#screenshots)
3. [How it works](#how-it-works)
4. [Architecture](#architecture)
5. [Run on macOS](#run-on-macos)
6. [Run on Windows / Linux](#run-on-windows--linux)
7. [Repository layout](#repository-layout)
8. [Environment variables](#environment-variables)
9. [Data and the synthetic sample](#data-and-the-synthetic-sample)
10. [Model and metrics](#model-and-metrics)
11. [Talk-to-Data agent](#talk-to-data-agent)
12. [Decision rules](#decision-rules)
13. [Five EDA insights](#five-eda-insights)
14. [Design notes](#design-notes)
15. [Troubleshooting](#troubleshooting)
16. [Limitations](#limitations)


## What it does

The five things a credit officer typically wants:

| # | Need | What this app does |
|---|---|---|
| 1 | Understand the loan portfolio | EDA dashboard with default rates split by demographic, financial, and credit-history dimensions. |
| 2 | Score a new applicant fast | LightGBM model returns probability of default + Low/Medium/High band + Approve/Review/Reject decision. |
| 3 | Justify the score | SHAP TreeExplainer gives exact feature contributions per applicant + a global driver chart. |
| 4 | Codify the model into a written policy | Decision-tree rule extractor outputs IF-THEN rules with support, default rate, and lift. |
| 5 | Ad-hoc questions without writing SQL | Chatbot that turns English into safe SELECT, runs it, and summarises the rows. |

All five are wired into a single Streamlit app, packaged in Docker so a
reviewer can run it with one command.


## Screenshots

| Section | Screenshot |
|---|---|
| Overview / KPIs | <img src="documents/screenshots/01_overview.png" width="600" /> |
| Exploratory Data Analysis | <img src="documents/screenshots/02_eda.png" width="600" /> |
| Risk Prediction (form + result) | <img src="documents/screenshots/03_predict.png" width="600" /> |
| Explainability (SHAP) | <img src="documents/screenshots/04_explain.png" width="600" /> |
| Decision Rules | <img src="documents/screenshots/05_rules.png" width="600" /> |
| Talk-to-Data Chatbot | <img src="documents/screenshots/06_chatbot.png" width="600" /> |


## How it works

**Data ingestion** (`src/data/`)

The loader picks whichever CSV is available, in this order:

```
data/raw/application_train.csv      # real Kaggle file (drop it in if you have it)
data/sample/application_train_sample.csv   # synthetic 10k rows (committed)
```

The chosen file gets ingested into `data/processed/credit_risk.db` (SQLite,
table `applications`). The DB is the single source of truth for both the
model and the chatbot, which means whatever the chatbot answers about is
exactly what the model was trained on.

**Training** (`src/ml/train.py`)

```
SQLite -> load_dataframe -> add_engineered_features -> ColumnTransformer
       -> LightGBMClassifier (scale_pos_weight handles 8% imbalance)
       -> joblib.dump(models/lgbm_model.joblib)
       -> metrics.json + classification_report.txt
```

**Inference** (`src/ml/predict.py`)

The UI form values are wrapped in a one-row DataFrame, run through the
same preprocessor that was fit during training, and scored. The
probability becomes a band:

```
P(default) <  0.20  ->  Low     ->  Approve
P(default) <  0.50  ->  Medium  ->  Review
P(default) >= 0.50  ->  High    ->  Reject
```

Thresholds are configurable in `.env`.

**Explainability** (`src/ml/explain.py`)

Per-applicant: a `shap.TreeExplainer` returns Shapley values for every
feature for that single applicant. The UI ranks them by absolute value
and labels each as "increases risk" or "decreases risk".

Global: same explainer applied to a 400-row sample, then mean |SHAP|
chart.

**Rules** (`src/rules/derive.py`)

A depth-4 decision tree is fit on the same engineered features as
LightGBM. Walking each leaf produces one IF-THEN rule with support
(% of population), default rate inside the leaf, and lift versus the
8% base rate.

**Talk-to-Data** (`src/llm/`)

```
question -> build_messages (pinned schema + 4 few-shots)
         -> LLM (OpenAI / Groq / Gemini, picks whichever key is set)
         -> parse JSON {"sql": "..."}
         -> sql_safety.validate_and_harden (single SELECT, table allowlist,
            blocked keywords, LIMIT cap)
         -> read-only SQLite execute
         -> 2nd LLM call grounds the summary in the actual rows
         -> Streamlit shows answer + SQL + rows
```

If no LLM key is set, the agent uses a deterministic regex parser that
covers nine canonical questions, so the demo still works offline.


## Architecture

```
+-------------------------- Streamlit UI (app/streamlit_app.py) --------------------------+
|   Overview | EDA | Risk Prediction | Explainability | Decision Rules | Chatbot          |
+----------+--------------+------------+---------------+----------------+------------------+
           |              |            |               |                |
           |              |     +------+------+ +------+------+ +-------+--------+
           |              |     |   src/ml    | |  src/rules  | |    src/llm     |
           |              |     |  train      | |   derive    | |   client       |
           |              |     |  predict    | |             | |   prompts      |
           |              |     |  explain    | |             | |   nl_to_sql    |
           |              |     +------+------+ +------+------+ +-------+--------+
           |              |            |               |                |
           v              v            v               v                v
       +-------------------------------------------------------------------+
       |  src/data  (schema, loader, sample_generator, preprocess)         |
       +------------------------------+------------------------------------+
                                      v
                  +-------------------------------------+
                  |  data/processed/credit_risk.db      |
                  |  (SQLite -- single source of truth) |
                  +-----------------+-------------------+
                                    ^
                                    |  ingested from
              +---------------------+----------------------------+
              |  data/raw/application_train.csv  (real Kaggle)   |
              |  -- or --                                        |
              |  data/sample/application_train_sample.csv        |
              |  (10k synthetic rows, committed)                 |
              +--------------------------------------------------+
```


## Run on macOS

Tested on macOS 13 / 14, both Apple Silicon and Intel. MacBook Pro 13" is
fine. Whole thing fits in under 2 GB of RAM.

### One-time setup

```bash
# Homebrew if you don't already have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# git, Python 3.11, Docker Desktop
brew install git python@3.11
brew install --cask docker
```

After Docker installs, **launch Docker Desktop once from /Applications**
and wait for the whale icon in the menu bar to become solid.

### Path A: Docker (easiest)

```bash
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

cp .env.example .env
# Optionally edit .env and paste ONE of:
#   OPENAI_API_KEY=sk-...
#   GROQ_API_KEY=gsk_...   (free tier, easiest)
#   GEMINI_API_KEY=...

docker compose up --build
```

When you see `URL: http://0.0.0.0:8501` in the terminal, open
<http://localhost:8501> in Safari or Chrome.

To stop: Ctrl+C, then `docker compose down`.

### Path B: native Python (no Docker)

```bash
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env

python scripts/build_db.py
python scripts/train_model.py
python scripts/derive_rules.py

streamlit run app/streamlit_app.py
```

Browser opens automatically at <http://localhost:8501>.

### Verify

```bash
pytest -q
```

Should print `9 passed`.


## Run on Windows / Linux

Same commands, with one substitution:

| Action | Windows PowerShell | macOS / Linux |
|---|---|---|
| Activate venv | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Set env var inline | `$env:GROQ_API_KEY="gsk_..."` | `export GROQ_API_KEY="gsk_..."` |

Everything else (Docker, `streamlit run`, `pytest`) is identical.


## Repository layout

```
credit-risk-platform/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                                 # drop Kaggle CSVs here (gitignored)
│   ├── sample/
│   │   └── application_train_sample.csv     # 10k synthetic rows
│   └── processed/                           # auto-built SQLite DB
├── documents/
│   ├── presentation.pdf
│   ├── presentation.pptx
│   └── screenshots/
├── models/
│   ├── lgbm_model.joblib
│   ├── metrics.json
│   ├── classification_report.txt
│   └── rules.json
├── scripts/
│   ├── generate_sample.py
│   ├── build_db.py
│   ├── train_model.py
│   ├── derive_rules.py
│   ├── download_kaggle.py
│   ├── capture_screenshots.py
│   └── build_presentation.py
├── src/
│   ├── config.py
│   ├── data/      # schema, loader, sample_generator, preprocess
│   ├── ml/        # train, predict, explain
│   ├── rules/     # derive
│   ├── llm/       # client, prompts, nl_to_sql
│   └── utils/     # sql_safety
├── tests/
│   ├── test_sql_safety.py
│   └── test_predict.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```


## Environment variables

See `.env.example`. The only thing you actually need is one of the
three LLM keys (free tier works for all three):

```
OPENAI_API_KEY=sk-...        # https://platform.openai.com/api-keys
GROQ_API_KEY=gsk_...         # https://console.groq.com/keys (free)
GEMINI_API_KEY=...           # https://aistudio.google.com/apikey (free)
```

Provider priority: OpenAI -> Groq -> Gemini -> deterministic fallback.

If no key is set, the chatbot still works -- it routes to a regex
parser that handles nine canonical questions.


## Data and the synthetic sample

The Kaggle Home Credit dataset is around 2.7 GB unzipped, with
`application_train.csv` alone at 286 MB. GitHub has a hard 100 MB
per-file limit, so we can't ship the real CSV in the repo.

The repo ships a 10 000-row synthetic sample that mirrors the real
dataset's column schema and overall ~8% default rate. This means:

- The platform runs end-to-end out of the box, no downloads required.
- Screenshots, the deck, and the unit tests all use this sample.
- The reported metrics (ROC-AUC ~0.90, KS ~0.66) come from this sample.

To use the real Kaggle data, download it locally:

```bash
# Option 1 -- the helper script (needs ~/.kaggle/kaggle.json)
python scripts/download_kaggle.py --files application_train.csv

# Option 2 -- manual: download from kaggle.com and save to data/raw/

# Either way, rebuild after:
python scripts/build_db.py
python scripts/train_model.py
python scripts/derive_rules.py
```

On the real data expect ROC-AUC around 0.74-0.78 (real-world data is
noisier than the calibrated synthetic sample).


## Model and metrics

| Choice | Why |
|---|---|
| LightGBM | Strong default performance on tabular credit data, handles mixed types after one-hot, trains in seconds. |
| `scale_pos_weight = neg/pos` | Better than naive over-/under-sampling on this data. |
| 80/20 stratified split | Keeps 8% default rate in both folds. |
| ROC-AUC primary metric | Kaggle competition metric, standard in credit risk. |
| KS, PR-AUC, F1 | KS is the canonical banking metric, PR-AUC is robust to imbalance. |
| Threshold = max(Youden's J) | Principled probability -> decision conversion. |
| Bands: Low <0.20, Medium <0.50, High >=0.50 | Tunable in `.env`. |

Reference numbers on the bundled sample:

| Metric | Value |
|---|---|
| ROC-AUC | 0.895 |
| PR-AUC | 0.511 |
| KS | 0.657 |
| F1 @ optimal threshold | 0.399 |
| Default rate | 8.0% |
| Train / val rows | 8 000 / 2 000 |


## Talk-to-Data agent

How we keep it from hallucinating SQL or numbers:

1. **Pinned schema** in the system prompt. No DB introspection per turn.
2. **JSON-only output**: `{"sql": "..."}`. We parse JSON, no fences.
3. **Few-shot kept tiny** (4 examples). Anchors style without bloating tokens.
4. **Temperature 0** + 256 max output tokens.
5. **Static SQL guard** (`src/utils/sql_safety.py`): single SELECT/WITH
   only, table allowlist, blocked keywords (INSERT/DROP/ATTACH/PRAGMA/...),
   LIMIT 200 injected if missing.
6. **Read-only SQLite** connection (URI mode `ro`).
7. **Result-grounded summary**: the second LLM call is told to use ONLY
   the rows we just fetched. Forbids invented numbers.
8. **Hard fallback**: regex intent parser handles nine canonical
   questions if no key is set or the LLM fails.

Working sample queries:

| # | Question | Returns |
|---|---|---|
| 1 | How many applicants in total? | one-row count |
| 2 | What is the overall default rate? | % defaulted |
| 3 | Default rate by education level | one row per education level |
| 4 | Top occupations by default rate | top 10 occupations (>=50 clients) |
| 5 | Average income for defaulters vs non-defaulters | 2 rows by TARGET |
| 6 | Default rate by gender | F vs M |
| 7 | Default rate by housing type | one row per housing type |
| 8 | Income distribution range | min / avg / max |
| 9 | Credit-to-income ratio for defaulters vs non | 2 rows by TARGET |


## Decision rules

A depth-4 decision tree with `min_samples_leaf = 2%` of the population.
Each leaf becomes one IF-THEN rule:

```
Rule 13: IF CREDIT_INCOME_RATIO > 5.19
        AND ANNUITY_INCOME_RATIO > 0.46
        AND EXT_SOURCE_MEAN <= 0.47
        AND NAME_EDUCATION_TYPE != 'Higher education'
        THEN risk = High  (support 6.9%, default rate 88.3%, lift 11.04x, n=686)
```

Rules ranked by `|leaf_default_rate - base_rate| * support` so the rules
that move the most population the most show up first.


## Five EDA insights

1. **External credit scores dominate.** `EXT_SOURCE_*` are the strongest
   single predictors; high scores cut the default rate by roughly 5x.
2. **Credit-to-income ratio matters more than absolute income.**
   Defaulters show meaningfully higher loan-to-income ratios.
3. **Education is a strong demographic signal.** Higher-education and
   Academic-degree clients default at noticeably lower rates.
4. **Age skew.** Younger applicants (<=30y) carry higher default
   probability; the curve flattens after about 45.
5. **Employment stability protects.** Longer tenure lowers default rate;
   the unemployed sentinel `DAYS_EMPLOYED = 365243` MUST become NaN or
   it dominates the signal.


## Design notes

- Streamlit instead of FastAPI + React: single process, fewer moving parts.
- SQLite as the data plane: zero config, file-based, same Python.
- Multi-provider LLM client: evaluators have any of OpenAI / Groq /
  Gemini, we autodetect.
- Deterministic chatbot fallback so the demo never fails offline.
- SHAP TreeExplainer for exact (not approximated) Shapley values.
- Pre-train the model inside the Docker image so the first request
  isn't slow.


## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose: command not found` | Install / launch Docker Desktop. Whale icon must be solid. |
| `port 8501 already in use` | `lsof -i :8501` then `kill <pid>`. Or change the port in `docker-compose.yml`. |
| OpenAI/Groq/Gemini errors | Wrong key or rate limited. Chatbot auto-falls-back to the deterministic parser. |
| LightGBM import error on Apple Silicon | `brew install libomp`, then `pip install -r requirements.txt`. |
| `Model artifact not found` | Run `python scripts/train_model.py`. Docker handles this at build time. |
| Charts not rendering | Hard refresh (`Cmd+Shift+R` / `Ctrl+Shift+R`). |
| Tests fail with `Model artifact missing` | Expected on a fresh clone. `pytest` skips that test until you train. |
| Want a clean slate | `rm -rf data/processed/* models/*` then re-run `build_db.py` + `train_model.py`. |


## Limitations

- Adding `bureau` and `previous_application` aggregations would lift AUC
  by another ~0.03 on the real Kaggle data.
- No drift monitoring (Evidently / WhyLogs would be the obvious next step).
- Single-tenant SQLite. Switch to Postgres for multi-user concurrency.
- No UI auth. Behind a reverse proxy in production.
- LLM cost cap is per-call only; daily quotas would need Redis-based
  rate limiting.

---

Author: Ashwani Dhayal
Submitted: May 2026
