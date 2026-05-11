"""Prediction schema and logic."""

from typing import Literal
from pydantic import BaseModel, Field

from src.model import get_model, get_scaler, get_feature_names


class CreditApplication(BaseModel):
    """Input schema for a credit application."""

    income: float = Field(..., description="Annual income in ZAR")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio (0.0-1.0)")
    loan_history_count: int = Field(..., description="Number of prior loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        ..., description="Home ownership status"
    )
    verified_income: int = Field(..., description="Income verified (0 or 1)")

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


class PredictionResult(BaseModel):
    """Output schema for a prediction."""

    approved: bool
    default_probability: float
    risk_band: Literal["low", "medium", "high"]


def make_prediction(application: CreditApplication) -> PredictionResult:
    """
    Run inference on a credit application.

    Args:
        application: Validated application features.

    Returns:
        PredictionResult with approval, probability, and risk band.
    """
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    feature_vector = [
        application.credit_score,
        application.income,
        application.debt_to_income,
        application.employment_years,
        0,  # loan_amount placeholder
        0,  # interest_rate placeholder
        application.verified_income,
        application.loan_history_count,
        0,  # delinquency_2yrs placeholder
        0,  # loan_purpose_business
        0,  # loan_purpose_debt_consolidation
        0,  # loan_purpose_home_improvement
        0,  # loan_purpose_major_purchase
        1 if application.home_ownership == "other" else 0,  # loan_purpose_other
        1 if application.home_ownership == "mortgage" else 0,  # home_ownership_MORTGAGE
        0,  # home_ownership_OTHER
        1 if application.home_ownership == "own" else 0,  # home_ownership_OWN
        1 if application.home_ownership == "rent" else 0,  # home_ownership_RENT
    ]

    X = scaler.transform([feature_vector])
    prob = model.predict_proba(X)[0, 1]

    if prob < 0.15:
        risk_band = "low"
    elif prob <= 0.35:
        risk_band = "medium"
    else:
        risk_band = "high"

    approved = prob < 0.35

    return PredictionResult(
        approved=approved,
        default_probability=round(float(prob), 4),
        risk_band=risk_band,
    )