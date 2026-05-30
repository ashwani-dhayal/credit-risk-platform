"""SHAP explanations for individual predictions and global drivers.

TreeExplainer gives exact Shapley values for tree models, which is what
LightGBM is. No sampling, no approximation.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from src.data.preprocess import add_engineered_features
from src.ml.predict import load_artifact


@dataclass
class FeatureContribution:
    feature: str
    value: object
    shap_value: float
    direction: str  # "increases risk" or "decreases risk"


@lru_cache(maxsize=1)
def _explainer():
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    return shap.TreeExplainer(pipeline.named_steps["classifier"])


def _transform(df):
    """Run the same preprocessor used at training time."""
    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    cols = artifact["feature_columns"]

    work = add_engineered_features(df.copy())
    for col in cols:
        if col not in work.columns:
            work[col] = np.nan

    X = pipeline.named_steps["preprocessor"].transform(work[cols])
    return np.asarray(X), artifact["feature_names_out"]


def _normalise_shap(sv):
    """LightGBM TreeExplainer output shape varies by SHAP version."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    return np.asarray(sv)


def explain_one(applicant, top_n=8):
    """Top-N SHAP contributions for a single applicant."""
    df = pd.DataFrame([applicant])
    X, feat_names = _transform(df)

    sv = _normalise_shap(_explainer().shap_values(X)).reshape(-1)
    pairs = list(zip(feat_names, sv, X.reshape(-1).tolist()))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    out = []
    for name, shap_val, value in pairs[:top_n]:
        if isinstance(value, float):
            value = round(value, 3)
        out.append(FeatureContribution(
            feature=name,
            value=value,
            shap_value=round(float(shap_val), 4),
            direction=("increases risk" if shap_val > 0 else "decreases risk"),
        ))
    return out


def global_importance(sample_size=500):
    """Mean |SHAP| across a random sample. Used by the global driver chart."""
    from src.data.loader import load_dataframe

    df = load_dataframe(nrows=sample_size)
    X, feat_names = _transform(df)
    sv = _normalise_shap(_explainer().shap_values(X))
    mean_abs = np.abs(sv).mean(axis=0)

    out = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
