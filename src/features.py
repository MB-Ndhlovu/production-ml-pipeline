"""
Feature engineering — must mirror the training pipeline exactly.

The model expects 18 features in this order:
  ['credit_score', 'annual_income', 'debt_to_income', 'employment_years',
   'loan_amount', 'interest_rate', 'verified_income', 'num_credit_lines',
   'delinquency_2yrs', 'loan_purpose_business', 'loan_purpose_debt_consolidation',
   'loan_purpose_home_improvement', 'loan_purpose_major_purchase',
   'loan_purpose_other', 'home_ownership_MORTGAGE', 'home_ownership_OTHER',
   'home_ownership_OWN', 'home_ownership_RENT']
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Loan purpose categories (from training data)
LOAN_PURPOSE_CATEGORIES = [
    "business", "debt_consolidation", "home_improvement", "major_purchase", "other"
]

# Home ownership categories (from training data)
HOME_OWNERSHIP_CATEGORIES = ["MORTGAGE", "OTHER", "OWN", "RENT"]


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ratio features."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["balance_to_income"] = 0  # not available in API input, set to 0
    df["credit_utilization"] = 0  # not available, set to 0
    df["debt_burden"] = 0  # not available, set to 0
    return df


def encode_categorical(df: pd.DataFrame, encoders: dict = None) -> tuple:
    """Label-encode home_ownership and one-hot encode loan_purpose."""
    df = df.copy()
    fit = encoders is None
    encoders = encoders or {}

    # home_ownership — label encode (0,1,2,3)
    if fit:
        encoders["home_ownership"] = LabelEncoder()
        encoders["home_ownership"].fit(HOME_OWNERSHIP_CATEGORIES)

    # map unknown categories to first class
    df["home_ownership_enc"] = df["home_ownership"].apply(
        lambda x: x if x in HOME_OWNERSHIP_CATEGORIES else HOME_OWNERSHIP_CATEGORIES[0]
    )
    df["home_ownership_enc"] = encoders["home_ownership"].transform(df["home_ownership_enc"])

    # one-hot encode loan_purpose
    for cat in LOAN_PURPOSE_CATEGORIES:
        df[f"loan_purpose_{cat}"] = (df.get("loan_purpose", "other") == cat).astype(float)

    # one-hot encode home_ownership
    for cat in HOME_OWNERSHIP_CATEGORIES:
        df[f"home_ownership_{cat}"] = (df["home_ownership"] == cat).astype(float)

    return df, encoders


def build_feature_vector(
    income: float,
    credit_score: int,
    employment_years: int,
    debt_to_income: float,
    loan_history_count: int,
    age: int,
    home_ownership: str,
    verified_income: int,
    loan_amount: float = 15000,  # default
    interest_rate: float = 0.14,  # default
    num_credit_lines: int = 5,  # default
    delinquency_2yrs: int = 0,  # default
    loan_purpose: str = "other",  # default
) -> np.ndarray:
    """
    Build the 18-element feature vector for the scaler.

    Mirrors the preprocessing from the training pipeline.
    """
    df = pd.DataFrame([{
        "annual_income": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_history_count": loan_history_count,
        "age": age,
        "home_ownership": home_ownership.upper(),
        "verified_income": verified_income,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "num_credit_lines": num_credit_lines,
        "delinquency_2yrs": delinquency_2yrs,
        "loan_purpose": loan_purpose,
    }])

    df = derive_features(df)
    df, _ = encode_categorical(df)

    # Feature order MUST match feature_names order
    feature_order = [
        "credit_score", "annual_income", "debt_to_income", "employment_years",
        "loan_amount", "interest_rate", "verified_income", "num_credit_lines",
        "delinquency_2yrs", "loan_purpose_business", "loan_purpose_debt_consolidation",
        "loan_purpose_home_improvement", "loan_purpose_major_purchase",
        "loan_purpose_other", "home_ownership_MORTGAGE", "home_ownership_OTHER",
        "home_ownership_OWN", "home_ownership_RENT",
    ]

    return df[feature_order].values.astype(float)