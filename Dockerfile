# syntax=docker/dockerfile:1.7
# ---------- Builder stage: install deps in a layer that can be cached ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs libgomp at build time for the prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .
RUN pip install --prefix=/install/deps -r requirements.txt

# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1 is required at runtime by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# Pre-train the model at build time so the container is "ready to serve"
# the moment it starts (no first-request latency, no surprise crashes).
# Uses the bundled synthetic sample if no Kaggle CSV is present.
RUN python scripts/build_db.py \
 && python scripts/train_model.py \
 && python scripts/derive_rules.py

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
