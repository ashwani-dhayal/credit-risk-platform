"""Feature engineering + sklearn preprocessing pipeline.

We split this into two stages so the same transformations are applied at
training time AND at single-row inference time from the UI:

1. `add_engineered_features(df)`  — pure pandas, deterministic.
2. `build_preprocessor()`         — sklearn ColumnTransformer that handles
                                     missing values + categorical encoding.

The fitted preprocessor is persisted alongside the model so the UI form
inputs flow through the exact same transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.schema import CATEGORICAL_COLUMNS, ENGINEERED_COLUMNS, NUMERIC_COLUMNS

# Sentinel that means "unemployed" in the Kaggle data. Convert to NaN so
# imputation handles it instead of letting it dominate as 365243 days.
DAYS_EMPLOYED_SENTINEL = 365243


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with engineered features and cleaned sentinels."""
    out = df.copy()

    # Replace the unemployed sentinel before deriving features.
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_SENTINEL, np.nan
        )

    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).round(2)
    if "DAYS_EMPLOYED" in out.columns:
        out["EMPLOYMENT_YEARS"] = (-out["DAYS_EMPLOYED"] / 365.25).round(2)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO"] = (
            out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = (
            out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(out.columns):
        out["CREDIT_TERM"] = (
            out["AMT_CREDIT"] / out["AMT_ANNUITY"].replace(0, np.nan)
        )
    if {"EMPLOYMENT_YEARS", "AGE_YEARS"}.issubset(out.columns):
        out["EMPLOYMENT_AGE_RATIO"] = (
            out["EMPLOYMENT_YEARS"] / out["AGE_YEARS"].replace(0, np.nan)
        )
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)

    return out


def feature_columns() -> list[str]:
    """Final list of features fed to the model after engineering."""
    return NUMERIC_COLUMNS + ENGINEERED_COLUMNS + CATEGORICAL_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """sklearn pipeline: median-impute numerics, most-frequent + OHE categoricals."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False so SHAP & LightGBM see a dense matrix.
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS + ENGINEERED_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return the column names produced by a fitted preprocessor."""
    return list(preprocessor.get_feature_names_out())
