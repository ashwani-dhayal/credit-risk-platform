"""Inference helpers: load the pickled pipeline, score one or many."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.data.preprocess import add_engineered_features


@dataclass
class Prediction:
    probability_default: float
    risk_band: str        # Low / Medium / High
    decision: str         # Approve / Review / Reject
    threshold: float
    used_threshold: bool


@lru_cache(maxsize=1)
def load_artifact(model_path=None):
    path = Path(model_path) if model_path else SETTINGS.model_path
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run `python scripts/train_model.py` first."
        )
    return joblib.load(path)


def _band(p):
    if p < SETTINGS.risk_low_max:
        return "Low"
    if p < SETTINGS.risk_medium_max:
        return "Medium"
    return "High"


_DECISION = {"Low": "Approve", "Medium": "Review", "High": "Reject"}


def predict_one(applicant):
    """Score a single applicant supplied as a dict from the UI form."""
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    cols = artifact["feature_columns"]

    df = pd.DataFrame([applicant])
    df = add_engineered_features(df)

    # Backfill any columns the model needs but the UI didn't provide.
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols]

    proba = float(pipeline.predict_proba(df)[0, 1])
    band = _band(proba)
    return Prediction(
        probability_default=round(proba, 4),
        risk_band=band,
        decision=_DECISION[band],
        threshold=float(artifact.get("optimal_threshold", 0.5)),
        used_threshold=False,
    )


def _load_model_and_preprocessor():
    """Return (model, preprocessor, feature_columns) for external use."""
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    cols = artifact["feature_columns"]
    model = pipeline.named_steps.get("classifier") or pipeline.steps[-1][1]
    preprocessor = pipeline.named_steps.get("preprocessor") or pipeline.steps[0][1]
    return model, preprocessor, cols


def predict_batch(df):
    """Score a DataFrame of applicants. Returns df with proba + band + decision."""
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
    out["risk_band"] = [_band(p) for p in proba]
    out["decision"] = [_DECISION[b] for b in out["risk_band"]]
    return out
