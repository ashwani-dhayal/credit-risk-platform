"""Smoke test for the prediction pipeline (requires a trained model)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SETTINGS  # noqa: E402


@pytest.mark.skipif(
    not SETTINGS.model_path.exists(),
    reason="Model artifact missing; run scripts/train_model.py first.",
)
def test_predict_one_returns_band():
    from src.ml.predict import predict_one
    sample = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 180_000,
        "AMT_CREDIT": 600_000,
        "AMT_ANNUITY": 30_000,
        "AMT_GOODS_PRICE": 540_000,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -35 * 365,
        "DAYS_EMPLOYED": -5 * 365,
        "OCCUPATION_TYPE": "Core staff",
        "CNT_FAM_MEMBERS": 2.0,
        "REGION_RATING_CLIENT": 2,
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": 0.55,
        "EXT_SOURCE_3": 0.50,
    }
    pred = predict_one(sample)
    assert 0.0 <= pred.probability_default <= 1.0
    assert pred.risk_band in {"Low", "Medium", "High"}
    assert pred.decision in {"Approve", "Review", "Reject"}
