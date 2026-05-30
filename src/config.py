"""Centralised configuration loaded from environment variables.

All paths resolve relative to the project root so the same code works
both in `python scripts/...` and inside Docker (where the workdir is /app).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present (no-op in production / Docker if absent).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else val


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    db_path: Path = PROJECT_ROOT / _env("DB_PATH", "data/processed/credit_risk.db")
    model_path: Path = PROJECT_ROOT / _env("MODEL_PATH", "models/lgbm_model.joblib")
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    sample_dir: Path = PROJECT_ROOT / "data" / "sample"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    documents_dir: Path = PROJECT_ROOT / "documents"

    # ---- LLM ----
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    groq_api_key: str = _env("GROQ_API_KEY")
    groq_model: str = _env("GROQ_MODEL", "llama-3.1-8b-instant")
    gemini_api_key: str = _env("GEMINI_API_KEY")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-1.5-flash")
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 512)
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.0)
    llm_timeout_seconds: int = _env_int("LLM_TIMEOUT_SECONDS", 20)

    # ---- Risk bands ----
    risk_low_max: float = _env_float("RISK_LOW_MAX", 0.20)
    risk_medium_max: float = _env_float("RISK_MEDIUM_MAX", 0.50)

    @property
    def active_llm_provider(self) -> str:
        """Return the first configured provider, or 'fallback' if none."""
        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        if self.gemini_api_key:
            return "gemini"
        return "fallback"


SETTINGS = Settings()
