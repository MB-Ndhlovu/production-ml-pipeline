"""Prediction schema and logic."""
from enum import Enum
from pydantic import BaseModel, Field


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionInput(BaseModel):
    """Input features for credit scoring prediction."""

    income: float = Field(..., gt=0, description="Annual income in currency units")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: str = Field(..., description="Home ownership status: rent, own, mortgage")
    verified_income: int = Field(..., ge=0, le=1, description="Income verified: 0=no, 1=yes")


class PredictionOutput(BaseModel):
    """Prediction response."""

    approved: bool = Field(..., description="Whether credit is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted probability of default")
    risk_band: RiskBand = Field(..., description="Risk classification")


def compute_risk_band(probability: float) -> RiskBand:
    """Classify probability into risk band."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def make_prediction(features: dict) -> PredictionOutput:
    """Run prediction given raw features."""
    from src.model import predict_proba

    probability = predict_proba(features)
    band = compute_risk_band(probability)

    return PredictionOutput(
        approved=probability < 0.5,
        default_probability=round(probability, 4),
        risk_band=band,
    )