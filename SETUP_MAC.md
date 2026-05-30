# macOS setup guide

Tested on macOS 13 / 14, MacBook Pro 13" (Intel and Apple Silicon).
Total time: about 5 minutes with Docker, 3 minutes without.


## Path 1: Docker (no Python knowledge needed)

Open Terminal (Cmd+Space, type "Terminal").

### 1. Install prerequisites (one time)

```bash
# Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# git and Docker Desktop
brew install git
brew install --cask docker
```

After installing, launch Docker Desktop from /Applications. Wait until
the whale icon in the menu bar is solid (not animated). That means
the Docker engine is running.

### 2. Clone, configure, run

```bash
cd ~/Documents
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

cp .env.example .env
# Optionally: nano .env  -- paste ONE LLM key (Groq is free and easiest)

docker compose up --build
```

The first build takes about 3 minutes (downloads the Python 3.11 base
image, installs deps, pre-trains the model). After that, restarts take
seconds.

### 3. Open the app

When Terminal shows:

```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

Open <http://localhost:8501> in Safari or Chrome.

### 4. Stop the app

Ctrl+C in Terminal, then:

```bash
docker compose down
```


## Path 2: native Python (no Docker)

### 1. Install prerequisites

```bash
brew install git python@3.11
```

### 2. Clone and set up

```bash
cd ~/Documents
git clone https://github.com/ashwani-dhayal/credit-risk-platform.git
cd credit-risk-platform

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# nano .env  -- optional, paste GROQ_API_KEY=gsk_... or similar
```

### 3. Build artifacts and launch

```bash
python scripts/build_db.py
python scripts/train_model.py
python scripts/derive_rules.py
streamlit run app/streamlit_app.py
```

The browser opens automatically at <http://localhost:8501>.

### 4. Stop / clean up

Ctrl+C in Terminal. To leave the venv: `deactivate`.


## Smoke test

```bash
pytest -q
```

Expected output: `9 passed`.


## Common issues

| Symptom | Fix |
|---|---|
| `xcrun: error: invalid active developer path` | `xcode-select --install` (one time) |
| `port 8501 already in use` | `lsof -i :8501` -> note PID -> `kill <pid>` |
| LightGBM import fails on Apple Silicon | `brew install libomp` then re-run `pip install -r requirements.txt` |
| Docker says "engine is starting" forever | Quit Docker Desktop, restart Mac, launch again |
| Browser doesn't open | Open <http://localhost:8501> manually |
| `docker compose up` permission denied | Make sure Docker Desktop is running (whale icon visible) |
| `403` from `download_kaggle.py` | Visit <https://www.kaggle.com/competitions/home-credit-default-risk/rules> and click "I Understand and Accept" |


## Optional: Groq API key (free, 30 seconds)

The chatbot works without any key, but with a real LLM you get richer
answers.

1. Go to <https://console.groq.com/keys>
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key (starts with `gsk_`)
5. Edit `.env`: `GROQ_API_KEY=gsk_...`
6. Restart Streamlit


## What you should see

After 30 seconds on the Overview page:

- **Applicants:** 10,000
- **Default rate:** 8.00%
- **Features:** 28
- **Model ROC-AUC:** 0.895

Then click around the sidebar:

1. **EDA** -- distributions and default rate charts
2. **Risk Prediction** -- fill the form, click Predict
3. **Explainability** -- top SHAP contributors for the prediction
4. **Decision Rules** -- 12 derived IF-THEN rules with support and lift
5. **Talk-to-Data** -- click any sample question, then Ask
