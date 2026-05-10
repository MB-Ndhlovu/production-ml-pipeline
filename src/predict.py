"""Prediction logic and Pydantic schemas."""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class HomeOwnership(str, Enum):
    OWN = "own"
    RENT = "rent"
    MORTGAGE = "mortgage"
    OTHER = "other"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CreditApplication(BaseModel):
    """Input schema for a credit application."""

    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
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


class PredictionResponse(BaseModel):
    """Output schema for a prediction."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Probability of default")
    risk_band: RiskBand = Field(..., description="Risk classification")


def get_risk_band(probability: float) -> RiskBand:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict_default_probability(application: CreditApplication, model, scaler, feature_names: list) -> float:
    """Compute the probability of default for a credit application."""
    import pandas as pd

    home_ownership_cols = [
        "home_ownership_MORTGAGE",
        "home_ownership_OTHER",
        "home_ownership_OWN",
        "home_ownership_RENT",
    ]
    loan_purpose_cols = [
        "loan_purpose_business",
        "loan_purpose_debt_consolidation",
        "loan_purpose_home_improvement",
        "loan_purpose_major_purchase",
        "loan_purpose_other",
    ]

    raw = {
        "credit_score": application.credit_score,
        "annual_income": application.income,
        "debt_to_income": application.debt_to_income,
        "employment_years": application.employment_years,
        "verified_income": application.verified_income,
    }

    df = pd.DataFrame([raw])
    df = df.reindex(columns=feature_names, fill_value=0)

    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0, 1]
    return float(prob)