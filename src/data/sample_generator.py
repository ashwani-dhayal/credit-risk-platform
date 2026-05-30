"""Generate a synthetic application_train sample for offline demos.

The full Kaggle CSV is 286 MB which doesn't fit in a normal git repo, so
we ship a 10k-row stand-in that has the same column names and a default
rate around 8%. The model trained on this is "demo quality"; for real
numbers drop the actual Kaggle CSV into data/raw/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CORE_COLUMNS

# Fixed seed so the bundled sample is byte-for-byte stable across runs.
RNG = np.random.default_rng(seed=42)


def _choice(options, probs, n):
    # numpy.random.choice barfs if probs don't sum to exactly 1, even by
    # 1e-9. Renormalising here saves us from chasing rounding bugs.
    arr = np.asarray(probs, dtype=float)
    arr = arr / arr.sum()
    return RNG.choice(options, size=n, p=arr)


def generate_sample(n_rows=10_000):
    n = n_rows

    # ---- demographics ----
    gender = _choice(["F", "M"], [0.66, 0.34], n)
    age_years = RNG.normal(loc=43, scale=11, size=n).clip(21, 70)
    days_birth = -(age_years * 365.25).astype(int)

    # ~18% unemployed/retired in real data; the rest get a tenure draw.
    employed_mask = RNG.random(n) > 0.18
    emp_years = np.where(
        employed_mask,
        RNG.gamma(shape=2.0, scale=4.0, size=n).clip(0.1, 40),
        0.0,
    )
    days_employed = np.where(
        employed_mask, -(emp_years * 365.25).astype(int), 365243
    )

    children = RNG.poisson(lam=0.4, size=n).clip(0, 6)
    fam_members = (children + RNG.choice([1, 2], size=n, p=[0.25, 0.75])).astype(float)

    # ---- categorical fields ----
    contract = _choice(["Cash loans", "Revolving loans"], [0.905, 0.095], n)
    own_car = _choice(["Y", "N"], [0.34, 0.66], n)
    own_realty = _choice(["Y", "N"], [0.69, 0.31], n)
    income_type = _choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
        [0.52, 0.23, 0.18, 0.06, 0.01],
        n,
    )
    education = _choice(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        [0.71, 0.24, 0.03, 0.015, 0.005],
        n,
    )
    family_status = _choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        [0.64, 0.15, 0.10, 0.06, 0.05],
        n,
    )
    housing = _choice(
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
        [0.88, 0.05, 0.04, 0.018, 0.008, 0.004],
        n,
    )
    occupation = _choice(
        [
            "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff",
            "Security staff", "Cleaning staff", "Cooking staff",
            "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
            "Realty agents", "Secretaries", "IT staff", "HR staff", np.nan,
        ],
        [
            0.16, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02,
            0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.31,
        ],
        n,
    )
    region_rating = _choice([1, 2, 3], [0.10, 0.74, 0.16], n).astype(int)

    # ---- money columns ----
    income = np.exp(RNG.normal(loc=11.9, scale=0.5, size=n)).round(-2).clip(25_000, 5_000_000)
    credit = np.exp(RNG.normal(loc=12.9, scale=0.6, size=n)).round(-2).clip(45_000, 4_000_000)
    annuity = (credit / RNG.uniform(8, 25, size=n)).round(-1)
    goods_price = (credit * RNG.uniform(0.85, 1.0, size=n)).round(-2)

    # External scores. ext1 has a real-world ~55% missingness pattern.
    ext1 = RNG.beta(2, 5, size=n)
    ext2 = RNG.beta(3, 4, size=n)
    ext3 = RNG.beta(2.5, 4.5, size=n)
    ext1_missing = RNG.random(n) < 0.55
    ext1[ext1_missing] = np.nan

    # ---- target generation ----
    # We build a logit out of the strongest real-world signals, then mark
    # the top 8% as defaulters via quantile threshold (instead of random
    # downsampling, which was destroying the signal).
    age = age_years
    credit_income = credit / np.maximum(income, 1)
    annuity_income = annuity / np.maximum(income, 1)
    ext_mean = np.nanmean(np.vstack([ext1, ext2, ext3]), axis=0)

    edu_risk = np.where(
        education == "Higher education", -0.6,
        np.where(education == "Academic degree", -0.9,
        np.where(education == "Incomplete higher", -0.15,
        np.where(education == "Secondary / secondary special", 0.15, 0.7)))
    )
    income_risk = np.where(
        income_type == "Unemployed", 2.0,
        np.where(income_type == "Pensioner", -0.5,
        np.where(income_type == "State servant", -0.3, 0.0))
    )
    housing_risk = np.where(
        housing == "Rented apartment", 0.4,
        np.where(housing == "With parents", 0.3, 0.0)
    )

    logit = (
        -3.4
        + 4.0 * (1 - ext_mean)            # external scores dominate
        + 0.55 * np.log1p(credit_income)
        + 0.9 * annuity_income
        + 0.025 * (45 - age)               # younger -> riskier
        + 0.08 * children
        + 0.55 * (region_rating - 2)
        + edu_risk
        + income_risk
        + housing_risk
        - 0.04 * np.log1p(emp_years)       # tenure protects
    )

    # Calibrate to ~8% default rate (matches Kaggle real ratio).
    target_rate = 0.08
    noisy_logit = logit + RNG.normal(0.0, 0.7, size=n)
    cutoff = np.quantile(noisy_logit, 1 - target_rate)
    target = (noisy_logit >= cutoff).astype(int)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100_001, 100_001 + n, dtype=np.int64),
        "TARGET": target,
        "NAME_CONTRACT_TYPE": contract,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children.astype(int),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": fam_members,
        "REGION_RATING_CLIENT": region_rating,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    })
    return df[CORE_COLUMNS]


def write_sample(out_path, n_rows=10_000):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample(n_rows=n_rows)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample/application_train_sample.csv")
    p.add_argument("--rows", type=int, default=10_000)
    args = p.parse_args()
    path = write_sample(Path(args.out), n_rows=args.rows)
    print(f"Wrote {args.rows:,} rows to {path}")
