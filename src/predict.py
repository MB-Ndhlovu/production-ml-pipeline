"""Prediction logic and Pydantic schemas for the credit scoring API."""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from .model import load_model, load_scaler, load_feature_names


class HomeOwnership(str, Enum):
    """Home ownership status."""
    OWN = "own"
    RENT = "rent"
    MORTGAGE = "mortgage"


class RiskBand(str, Enum):
    """Risk classification band."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionInput(BaseModel):
    """Schema for credit scoring prediction request."""

    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: float = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of loans in history")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0 or 1)")

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
                "verified_income": 1
            }
        }
    }


class PredictionOutput(BaseModel):
    """Schema for credit scoring prediction response."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted probability of default")
    risk_band: RiskBand = Field(..., description="Risk classification band")


def get_risk_band(probability: float) -> RiskBand:
    """Determine risk band from default probability.

    Args:
        probability: Predicted default probability (0-1)

    Returns:
        RiskBand: low (<0.15), medium (0.15-0.35), or high (>0.35)
    """
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict(input_data: PredictionInput) -> PredictionOutput:
    """Run credit scoring prediction on input features.

    Args:
        input_data: Validated prediction request

    Returns:
        PredictionOutput with approval decision, probability, and risk band
    """
    model = load_model()
    scaler = load_scaler()

    # Build feature vector matching model's expected 18 features
    # Feature order: credit_score, annual_income, debt_to_income, employment_years,
    # loan_amount, interest_rate, verified_income, num_credit_lines,
    # delinquency_2yrs, loan_purpose_*, home_ownership_*
    import numpy as np

    features = np.zeros(18)
    features[0] = input_data.credit_score                      # credit_score
    features[1] = input_data.income                             # annual_income
    features[2] = input_data.debt_to_income                     # debt_to_income
    features[3] = input_data.employment_years                    # employment_years
    features[4] = 0                                             # loan_amount (not provided)
    features[5] = 0                                             # interest_rate (not provided)
    features[6] = input_data.verified_income                    # verified_income
    features[7] = input_data.loan_history_count                 # num_credit_lines

    # home_ownership: OWN->2, RENT->3, MORTGAGE->0, OTHER->1
    ho = input_data.home_ownership.value.lower()
    if ho == "own":
        features[15] = 1
    elif ho == "rent":
        features[17] = 1
    elif ho == "mortgage":
        features[14] = 1
    else:
        features[16] = 1

    features_scaled = scaler.transform(features.reshape(1, -1))
    probability = float(model.predict_proba(features_scaled)[0, 1])
    risk_band = get_risk_band(probability)
    approved = probability < 0.35

    return PredictionOutput(
        approved=approved,
        default_probability=round(probability, 4),
        risk_band=risk_band,
    )
