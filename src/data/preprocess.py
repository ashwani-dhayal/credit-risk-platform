"""Feature engineering and the sklearn ColumnTransformer.

Two-stage setup so training and single-row UI inference share the exact
same transformations:

    add_engineered_features(df)  -> pandas, deterministic, idempotent
    build_preprocessor()         -> imputer + one-hot

The fitted preprocessor is pickled together with the model so the UI
form values flow through identically at predict time.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.schema import CATEGORICAL_COLUMNS, ENGINEERED_COLUMNS, NUMERIC_COLUMNS

# Magic value Kaggle uses for "unemployed". Has to become NaN before we
# compute any tenure feature, otherwise it dominates the signal.
DAYS_EMPLOYED_SENTINEL = 365243


def add_engineered_features(df):
    out = df.copy()

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

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                if c in out.columns]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)

    return out


def feature_columns():
    return NUMERIC_COLUMNS + ENGINEERED_COLUMNS + CATEGORICAL_COLUMNS


def build_preprocessor():
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    # sparse_output=False because SHAP and the dense LightGBM path are
    # both happier with a dense matrix.
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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


def get_feature_names_out(preprocessor):
    return list(preprocessor.get_feature_names_out())
