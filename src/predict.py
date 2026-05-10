"""Prediction logic and Pydantic schemas for credit scoring API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.model import get_feature_names, get_model, get_scaler


class HomeOwnership(str, Enum):
    OWN = "own"
    RENT = "rent"
    MORTGAGE = "mortgage"


class PredictInput(BaseModel):
    """Schema for POST /predict request body."""

    income: float = Field(..., gt=0, description="Annual income in ZAR")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., gt=17, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Housing status")
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0/1)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "income": 65000,
                "credit_score": 720,
                "employment_years": 5,
                "debt_to_income": 0.28,
                "loan_history_count": 2,
                "age": 34,
                "home_ownership": "rent",
                "verified_income": 1,
            }
        }
    }


class PredictOutput(BaseModel):
    """Schema for POST /predict response body."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted default probability")
    risk_band: Literal["low", "medium", "high"] = Field(..., description="Risk classification")


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Classify probability into risk bands."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    return "high"


def _build_feature_vector(input_data: PredictInput):
    """
    Build a 18-element feature vector matching the training feature order:
    ['credit_score', 'annual_income', 'debt_to_income', 'employment_years',
     'loan_amount', 'interest_rate', 'verified_income', 'num_credit_lines',
     'delinquency_2yrs', 'loan_purpose_business', 'loan_purpose_debt_consolidation',
     'loan_purpose_home_improvement', 'loan_purpose_major_purchase', 'loan_purpose_other',
     'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT']
    """
    home = input_data.home_ownership.value
    return [
        input_data.credit_score,       # credit_score
        input_data.income,              # annual_income
        input_data.debt_to_income,      # debt_to_income
        input_data.employment_years,   # employment_years
        0,                             # loan_amount (not provided)
        0,                             # interest_rate (not provided)
        input_data.verified_income,     # verified_income
        input_data.loan_history_count, # num_credit_lines
        0,                             # delinquency_2yrs (not provided)
        0,                             # loan_purpose_business
        0,                             # loan_purpose_debt_consolidation
        0,                             # loan_purpose_home_improvement
        0,                             # loan_purpose_major_purchase
        1,                             # loan_purpose_other (default)
        1 if home == "mortgage" else 0, # home_ownership_MORTGAGE
        0,                             # home_ownership_OTHER
        1 if home == "own" else 0,      # home_ownership_OWN
        1 if home == "rent" else 0,      # home_ownership_RENT
    ]


def predict(input_data: PredictInput) -> PredictOutput:
    """
    Run credit default prediction on a single applicant.

    Loads model + scaler on first call, then transforms features and returns
    the predicted default probability and derived fields.
    """
    model = get_model()
    scaler = get_scaler()

    feature_vector = _build_feature_vector(input_data)
    scaled = scaler.transform([feature_vector])
    probability = float(model.predict_proba(scaled)[0, 1])
    band = get_risk_band(probability)

    return PredictOutput(
        approved=probability < 0.5,
        default_probability=round(probability, 6),
        risk_band=band,
    )