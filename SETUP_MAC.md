# 🍎 macOS setup — paste this into Terminal

> Tested on **macOS 13 / 14**, **MacBook Pro 13″ (Intel & Apple Silicon)**.
> Estimated total time: **5 minutes** with Docker, **3 minutes** without.

---

## Path 1 — Docker (zero Python knowledge needed)

Open **Terminal** (⌘+Space → "Terminal"), then paste each block.

### 1. Install prerequisites (one time only)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install git and Docker Desktop
brew install git
brew install --cask docker
```

After Docker installs, **launch Docker Desktop from /Applications**. Wait
for the whale icon in the menu bar to become **solid** (not animated).
That means the Docker engine is running.

### 2. Clone, configure, run

```bash
cd ~/Documents
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

cp .env.example .env
# Optional: nano .env  → paste ONE LLM key (Groq is free & easiest)

docker compose up --build
```

First build takes ~3 minutes (downloads Python 3.11 base image + installs
deps + pre-trains the model). Subsequent starts take ~5 seconds.

### 3. Open the app

When Terminal shows:

```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

Open <http://localhost:8501> in Safari or Chrome.

### 4. Stop the app

Press **Ctrl+C** in Terminal, then:

```bash
docker compose down
```

---

## Path 2 — Native Python (no Docker)

### 1. Install prerequisites

```bash
brew install git python@3.11
```

### 2. Clone and set up

```bash
cd ~/Documents
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

# Create an isolated Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional LLM key
cp .env.example .env
# nano .env  → paste GROQ_API_KEY=gsk_... (or OPENAI / GEMINI)
```

### 3. Build artifacts and launch (~30 s)

```bash
python scripts/build_db.py
python scripts/train_model.py
python scripts/derive_rules.py
streamlit run app/streamlit_app.py
```

Streamlit auto-opens your browser at <http://localhost:8501>.

### 4. Stop / cleanup

Press **Ctrl+C** in Terminal. To deactivate the venv: `deactivate`.

---

## 🧪 Smoke-test that everything works

```bash
pytest -q
```

Expected output:

```
9 passed in 2.5s
```

---

## 🆘 Common Mac issues

| Symptom | Fix |
|---|---|
| `xcrun: error: invalid active developer path` | `xcode-select --install` (one-time). |
| `port 8501 already in use` | `lsof -i :8501` → note the PID → `kill <pid>`. |
| `LightGBM` import fails on Apple Silicon | `brew install libomp` then re-run `pip install -r requirements.txt`. |
| Docker says "engine is starting" forever | Quit Docker Desktop, restart Mac, launch Docker again. |
| Browser doesn't open automatically | Open <http://localhost:8501> manually. |
| `docker compose up` errors with permission denied | Make sure Docker Desktop is **running** (whale icon visible). |
| You get a `403` from `python scripts/download_kaggle.py` | Visit <https://www.kaggle.com/competitions/home-credit-default-risk/rules> and click "I Understand and Accept" once. |

---

## 🔁 Optional: get a Groq API key (free, 30 seconds)

The chatbot works without any key, but with a real LLM the answers are
much richer.

1. Visit <https://console.groq.com/keys>
2. Sign in with Google (free)
3. Click **Create API Key**
4. Copy the key (starts with `gsk_`)
5. Edit `.env`: `GROQ_API_KEY=gsk_...`
6. Restart Streamlit

---

## ✅ What you should see

After 30 seconds on the Overview page you'll see:

- **Applicants:** 10,000
- **Default rate:** 8.00%
- **Features:** 28
- **Model ROC-AUC:** 0.895

Then click around the sidebar:

1. **EDA** — Plotly charts of distributions and default rates.
2. **Risk Prediction** — fill the form, click 🚀 Predict.
3. **Explainability** — top SHAP contributors for the prediction you just made.
4. **Decision Rules** — 12 derived IF-THEN rules with support and lift.
5. **Talk-to-Data** — click any sample question, then 🔎 Ask.

That's the full demo.
