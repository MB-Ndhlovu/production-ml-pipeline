from pydantic import BaseModel
from typing import Literal

from src.model import get_model, get_scaler, get_feature_names


class PredictionInput(BaseModel):
    income: float
    credit_score: int
    employment_years: int
    debt_to_income: float
    loan_history_count: int
    age: int
    home_ownership: Literal["rent", "own", "mortgage", "other"]
    verified_income: int

    loan_amount: float = 0.0
    interest_rate: float = 0.0
    num_credit_lines: int = 0
    delinquency_2yrs: int = 0
    loan_purpose: Literal[
        "business", "debt_consolidation", "home_improvement", "major_purchase", "other"
    ] = "other"


def predict(input_data: PredictionInput) -> dict:
    """
    Make a credit default prediction from input features.

    Returns a dict with:
    - approved: bool
    - default_probability: float
    - risk_band: "low" | "medium" | "high"
    """
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    # One-hot encode home_ownership
    home_oh = {
        "MORTGAGE": 1 if input_data.home_ownership == "mortgage" else 0,
        "OTHER": 1 if input_data.home_ownership == "other" else 0,
        "OWN": 1 if input_data.home_ownership == "own" else 0,
        "RENT": 1 if input_data.home_ownership == "rent" else 0,
    }

    # One-hot encode loan_purpose
    purpose_oh = {
        "business": 1 if input_data.loan_purpose == "business" else 0,
        "debt_consolidation": 1 if input_data.loan_purpose == "debt_consolidation" else 0,
        "home_improvement": 1 if input_data.loan_purpose == "home_improvement" else 0,
        "major_purchase": 1 if input_data.loan_purpose == "major_purchase" else 0,
        "other": 1 if input_data.loan_purpose == "other" else 0,
    }

    raw_features = [
        input_data.credit_score,          # credit_score
        input_data.income,                # annual_income
        input_data.debt_to_income,         # debt_to_income
        input_data.employment_years,       # employment_years
        input_data.loan_amount,           # loan_amount
        input_data.interest_rate,         # interest_rate
        input_data.verified_income,       # verified_income
        input_data.num_credit_lines,      # num_credit_lines
        input_data.delinquency_2yrs,      # delinquency_2yrs
        purpose_oh["business"],           # loan_purpose_business
        purpose_oh["debt_consolidation"], # loan_purpose_debt_consolidation
        purpose_oh["home_improvement"],   # loan_purpose_home_improvement
        purpose_oh["major_purchase"],     # loan_purpose_major_purchase
        purpose_oh["other"],              # loan_purpose_other
        home_oh["MORTGAGE"],              # home_ownership_MORTGAGE
        home_oh["OTHER"],                 # home_ownership_OTHER
        home_oh["OWN"],                   # home_ownership_OWN
        home_oh["RENT"],                  # home_ownership_RENT
    ]

    feature_vector = scaler.transform([raw_features])
    prob = model.predict_proba(feature_vector)[0][1]

    # Determine risk band
    if prob < 0.15:
        risk_band = "low"
    elif prob <= 0.35:
        risk_band = "medium"
    else:
        risk_band = "high"

    approved = prob < 0.35

    return {
        "approved": bool(approved),
        "default_probability": round(float(prob), 4),
        "risk_band": risk_band,
    }