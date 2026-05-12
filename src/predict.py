"""Prediction schema and function for credit scoring."""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class HomeOwnership(str, Enum):
    OWN = "own"
    RENT = "rent"
    MORTGAGE = "mortgage"
    OTHER = "other"


class CreditScoreInput(BaseModel):
    """Input features for a credit score prediction."""

    income: float = Field(..., description="Annual income in ZAR")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., description="Income verified (0 or 1)")


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionOutput(BaseModel):
    """Output of the credit score prediction."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted probability of default")
    risk_band: RiskBand = Field(..., description="Risk classification band")


def get_risk_band(probability: float) -> RiskBand:
    """Classify probability into a risk band."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict(input_data: CreditScoreInput) -> PredictionOutput:
    """
    Run prediction on a single credit score input.

    Applies the scaler, computes the default probability from the model,
    and derives the approval decision and risk band.
    """
    from src.model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names: List[str] = get_feature_names()

    # Map input to feature vector using expected feature names
    raw_features = {
        "income": input_data.income,
        "credit_score": input_data.credit_score,
        "employment_years": input_data.employment_years,
        "debt_to_income": input_data.debt_to_income,
        "loan_history_count": input_data.loan_history_count,
        "age": input_data.age,
        "home_ownership": input_data.home_ownership.value,
        "verified_income": input_data.verified_income,
    }

    # Build ordered feature vector using exact names from feature_names
    import pandas as pd
    df = pd.DataFrame([raw_features])
    feature_vector = df.reindex(columns=feature_names, fill_value=0).values[0]

    # Scale and predict
    scaled = scaler.transform([feature_vector])
    probability = float(model.predict_proba(scaled)[0, 1])
    approved = probability < 0.35

    return PredictionOutput(
        approved=approved,
        default_probability=round(probability, 6),
        risk_band=get_risk_band(probability),
    )