"""Prediction schemas and logic for credit risk."""

from enum import Enum
from pydantic import BaseModel, Field


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionInput(BaseModel):
    """Schema for credit application features."""

    income: float = Field(..., description="Annual income in ZAR")
    credit_score: int = Field(..., description="Credit score (300–850)")
    employment_years: int = Field(..., description="Years in current employment")
    debt_to_income: float = Field(..., description="Debt-to-income ratio (0–1)")
    loan_history_count: int = Field(..., description="Number of previous loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: str = Field(..., description="Home ownership status: rent/own/mortgage")
    verified_income: int = Field(..., description="Income verified: 1=yes, 0=no")


class PredictionOutput(BaseModel):
    """Schema for prediction response."""

    approved: bool = Field(..., description="Loan approval decision")
    default_probability: float = Field(..., description="Predicted probability of default")
    risk_band: RiskBand = Field(..., description="Risk classification")


def get_risk_band(probability: float) -> RiskBand:
    """Classify probability into risk band."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH