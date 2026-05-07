"""Prediction utilities and Pydantic schemas."""
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from .model import load_model, load_scaler, load_feature_names

_model = None
_scaler = None
_feature_names = None


def get_model():
    global _model, _scaler, _feature_names
    if _model is None:
        _model = load_model()
        _scaler = load_scaler()
        _feature_names = load_feature_names()
    return _model, _scaler, _feature_names


class PredictionInput(BaseModel):
    """Input schema for credit scoring prediction."""

    income: float = Field(..., description="Annual income in ZAR")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., description="Number of past loans")
    age: int = Field(..., description="Age in years")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        default="rent", description="Home ownership status"
    )
    verified_income: int = Field(..., description="Income verified (0 or 1)")
    loan_amount: float = Field(default=10000, description="Loan amount in ZAR")
    interest_rate: float = Field(default=0.12, description="Annual interest rate as decimal (e.g. 0.12 for 12%)")
    num_credit_lines: int = Field(default=2, description="Number of active credit lines")
    delinquency_2yrs: int = Field(default=0, description="Delinquencies in past 2 years")
    loan_purpose: Literal[
        "business", "debt_consolidation", "home_improvement", "major_purchase", "other"
    ] = Field(default="other", description="Purpose of the loan")

    class Config:
        json_schema_extra = {
            "example": {
                "income": 65000,
                "credit_score": 720,
                "employment_years": 5,
                "debt_to_income": 0.28,
                "loan_history_count": 2,
                "age": 34,
                "home_ownership": "rent",
                "verified_income": 1,
                "loan_amount": 10000,
                "interest_rate": 0.12,
                "num_credit_lines": 2,
                "delinquency_2yrs": 0,
                "loan_purpose": "other",
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for credit scoring prediction."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., description="Predicted probability of default")
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk band based on default probability"
    )


# Feature order matches credit-scoring-pipeline training:
# ['credit_score', 'annual_income', 'debt_to_income', 'employment_years',
#  'loan_amount', 'interest_rate', 'verified_income', 'num_credit_lines',
#  'delinquency_2yrs', 'loan_purpose_business', 'loan_purpose_debt_consolidation',
#  'loan_purpose_home_improvement', 'loan_purpose_major_purchase', 'loan_purpose_other',
#  'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN',
#  'home_ownership_RENT']


def _build_feature_vector(input_data: PredictionInput) -> np.ndarray:
    """Build a 18-element feature vector matching the training feature order."""
    loan_purposes = ["business", "debt_consolidation", "home_improvement", "major_purchase", "other"]
    home_ownerships = ["mortgage", "other", "own", "rent"]

    features = [
        input_data.credit_score,        # credit_score
        input_data.income,              # annual_income
        input_data.debt_to_income,      # debt_to_income
        input_data.employment_years,    # employment_years
        input_data.loan_amount,         # loan_amount
        input_data.interest_rate,       # interest_rate
        input_data.verified_income,     # verified_income
        input_data.num_credit_lines,    # num_credit_lines
        input_data.delinquency_2yrs,    # delinquency_2yrs
        # loan purpose one-hot (5 values)
        1 if input_data.loan_purpose == "business" else 0,
        1 if input_data.loan_purpose == "debt_consolidation" else 0,
        1 if input_data.loan_purpose == "home_improvement" else 0,
        1 if input_data.loan_purpose == "major_purchase" else 0,
        1 if input_data.loan_purpose == "other" else 0,
        # home ownership one-hot (4 values)
        1 if input_data.home_ownership == "mortgage" else 0,
        1 if input_data.home_ownership == "other" else 0,
        1 if input_data.home_ownership == "own" else 0,
        1 if input_data.home_ownership == "rent" else 0,
    ]
    return np.array([features])


def predict_default(input_data: PredictionInput) -> PredictionOutput:
    """Predict credit default probability and return risk assessment."""
    model, scaler, feature_names = get_model()

    features = _build_feature_vector(input_data)
    scaled_features = scaler.transform(features)
    prob = model.predict_proba(scaled_features)[0][1]

    if prob < 0.15:
        band = "low"
        approved = True
    elif prob <= 0.35:
        band = "medium"
        approved = True
    else:
        band = "high"
        approved = False

    return PredictionOutput(
        approved=approved, default_probability=round(float(prob), 4), risk_band=band
    )