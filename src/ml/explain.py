"""SHAP-based explainability for individual predictions and global drivers.

We use SHAP TreeExplainer because LightGBM is a tree ensemble — this gives
exact, fast Shapley values and works without any black-box approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.data.preprocess import add_engineered_features
from src.ml.predict import load_artifact


@dataclass
class FeatureContribution:
    feature: str
    value: Any
    shap_value: float
    direction: str  # "increases risk" | "decreases risk"


@lru_cache(maxsize=1)
def _explainer():
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


def _transform(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    cols = artifact["feature_columns"]
    work = add_engineered_features(df.copy())
    for col in cols:
        if col not in work.columns:
            work[col] = np.nan
    X = pipeline.named_steps["preprocessor"].transform(work[cols])
    feat_names = artifact["feature_names_out"]
    return np.asarray(X), feat_names


def explain_one(
    applicant: dict[str, Any], top_n: int = 8
) -> list[FeatureContribution]:
    """Return the top-N SHAP contributions for a single applicant."""
    df = pd.DataFrame([applicant])
    X, feat_names = _transform(df)
    explainer = _explainer()
    sv = explainer.shap_values(X)
    # LightGBM binary classifier: shap returns array shape (1, n_features)
    # (older versions returned a list; handle both)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv).reshape(-1)

    pairs = list(zip(feat_names, sv, X.reshape(-1).tolist()))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    out: list[FeatureContribution] = []
    for name, shap_val, value in pairs[:top_n]:
        out.append(
            FeatureContribution(
                feature=name,
                value=_pretty(value),
                shap_value=round(float(shap_val), 4),
                direction="increases risk" if shap_val > 0 else "decreases risk",
            )
        )
    return out


def global_importance(sample_size: int = 500) -> pd.DataFrame:
    """Mean |SHAP| across a random sample (used for global driver chart)."""
    from src.data.loader import load_dataframe
    df = load_dataframe(nrows=sample_size)
    X, feat_names = _transform(df)
    explainer = _explainer()
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _pretty(v: Any) -> Any:
    try:
        if isinstance(v, float):
            return round(v, 3)
    except Exception:
        pass
    return v
