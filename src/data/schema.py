"""Column lists for the application_train table.

We don't keep all 122 Kaggle columns - just the 23 that move the model
and that an analyst is actually likely to ask about. Smaller schema =
smaller prompts, smaller DB, and a model that's still easy to read.
"""

# ---- columns we keep from application_train.csv ----
CORE_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
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

# Fields the user fills in the Streamlit form (subset of CORE_COLUMNS,
# minus the IDs and target).
UI_INPUT_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
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

NUMERIC_COLUMNS = [
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

# These get added by preprocess.add_engineered_features().
ENGINEERED_COLUMNS = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "EMPLOYMENT_AGE_RATIO",
    "EXT_SOURCE_MEAN",
]

# Short descriptions shown to the LLM (for SQL generation) and to the UI.
COLUMN_DESCRIPTIONS = {
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

TABLE_NAME = "applications"
