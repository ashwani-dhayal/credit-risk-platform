"""Inference utilities: load model, score one applicant, return risk band."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.data.preprocess import add_engineered_features, feature_columns


@dataclass
class Prediction:
    probability_default: float
    risk_band: str        # "Low" | "Medium" | "High"
    decision: str         # "Approve" | "Review" | "Reject"
    threshold: float      # the operating threshold used for `decision`
    used_threshold: bool  # True when `decision` came from threshold


@lru_cache(maxsize=1)
def load_artifact(model_path: Optional[str] = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else SETTINGS.model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run `python scripts/train_model.py` first."
        )
    return joblib.load(path)


def _classify_band(p: float) -> str:
    if p < SETTINGS.risk_low_max:
        return "Low"
    if p < SETTINGS.risk_medium_max:
        return "Medium"
    return "High"


def _decision_from_band(band: str) -> str:
    return {"Low": "Approve", "Medium": "Review", "High": "Reject"}[band]


def predict_one(applicant: dict[str, Any]) -> Prediction:
    """Score a single applicant supplied as a {column: value} dict from the UI."""
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    cols = artifact["feature_columns"]

    df = pd.DataFrame([applicant])
    df = add_engineered_features(df)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]

    proba = float(pipeline.predict_proba(df)[0, 1])
    band = _classify_band(proba)
    return Prediction(
        probability_default=round(proba, 4),
        risk_band=band,
        decision=_decision_from_band(band),
        threshold=float(artifact.get("optimal_threshold", 0.5)),
        used_threshold=False,
    )


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of applicants. Returns probability + band."""
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    cols = artifact["feature_columns"]

    work = add_engineered_features(df.copy())
    for col in cols:
        if col not in work.columns:
            work[col] = np.nan
    proba = pipeline.predict_proba(work[cols])[:, 1]
    out = df.copy()
    out["probability_default"] = proba
    out["risk_band"] = [_classify_band(p) for p in proba]
    out["decision"] = [_decision_from_band(b) for b in out["risk_band"]]
    return out
