"""Schema definitions for the Home Credit Default Risk dataset.

We model only the `application_train` columns we actively use. The full
Kaggle dataset has 122 columns; we keep the 18 that are most informative
for default prediction and that an analyst would realistically ask about.
This keeps the SQLite schema small, the prompts cheap, and the model
interpretable.
"""

from __future__ import annotations

from typing import Final

# Columns retained from application_train.csv
CORE_COLUMNS: Final[list[str]] = [
    "SK_ID_CURR",          # primary key
    "TARGET",              # 1 = default, 0 = repaid (only in train)
    "NAME_CONTRACT_TYPE",  # Cash loans / Revolving loans
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",          # negative integer (days before application)
    "DAYS_EMPLOYED",       # negative integer; 365243 sentinel = unemployed
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

# Columns the user enters via the UI form (subset, with friendly defaults)
UI_INPUT_COLUMNS: Final[list[str]] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLUMNS: Final[list[str]] = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

# Engineered features added in preprocess.add_engineered_features()
ENGINEERED_COLUMNS: Final[list[str]] = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "EMPLOYMENT_AGE_RATIO",
    "EXT_SOURCE_MEAN",
]

# Friendly column descriptions surfaced to the LLM and the UI.
COLUMN_DESCRIPTIONS: Final[dict[str, str]] = {
    "SK_ID_CURR": "Unique application ID",
    "TARGET": "1 if the client defaulted, 0 if repaid on time",
    "NAME_CONTRACT_TYPE": "Loan type (Cash loans or Revolving loans)",
    "CODE_GENDER": "Gender of the applicant (M / F / XNA)",
    "FLAG_OWN_CAR": "Y if the applicant owns a car, else N",
    "FLAG_OWN_REALTY": "Y if the applicant owns property, else N",
    "CNT_CHILDREN": "Number of children",
    "AMT_INCOME_TOTAL": "Annual income of the applicant",
    "AMT_CREDIT": "Total credit amount of the loan",
    "AMT_ANNUITY": "Loan annuity (yearly repayment)",
    "AMT_GOODS_PRICE": "Price of the goods for which the loan is given",
    "NAME_INCOME_TYPE": "Income source category",
    "NAME_EDUCATION_TYPE": "Highest education level",
    "NAME_FAMILY_STATUS": "Marital / family status",
    "NAME_HOUSING_TYPE": "Housing situation",
    "DAYS_BIRTH": "Age in days, negative (days before application)",
    "DAYS_EMPLOYED": "Days employed, negative; 365243 means unemployed",
    "OCCUPATION_TYPE": "Occupation category",
    "CNT_FAM_MEMBERS": "Number of family members",
    "REGION_RATING_CLIENT": "Region rating (1 best, 3 worst)",
    "EXT_SOURCE_1": "Normalised score from external data source 1 (0..1)",
    "EXT_SOURCE_2": "Normalised score from external data source 2 (0..1)",
    "EXT_SOURCE_3": "Normalised score from external data source 3 (0..1)",
    "AGE_YEARS": "Applicant age in years (engineered)",
    "EMPLOYMENT_YEARS": "Years employed (engineered)",
    "CREDIT_INCOME_RATIO": "Loan amount / annual income (engineered)",
    "ANNUITY_INCOME_RATIO": "Annuity / annual income (engineered)",
    "CREDIT_TERM": "Loan term in years = AMT_CREDIT / AMT_ANNUITY (engineered)",
    "EMPLOYMENT_AGE_RATIO": "Years employed / age (engineered)",
    "EXT_SOURCE_MEAN": "Mean of EXT_SOURCE_1/2/3 (engineered)",
}

TABLE_NAME: Final[str] = "applications"
