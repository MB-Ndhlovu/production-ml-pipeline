"""Prediction logic and Pydantic schemas for the credit scoring API."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from . import model as model_module


class RiskBand(str, Enum):
    """Risk classification bands based on default probability."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CreditApplication(BaseModel):
    """Schema for a credit application prediction request."""
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years in current employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage"] = Field(
        ..., description="Home ownership status"
    )
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
                "verified_income": 1,
            }
        }
    }


class PredictionResult(BaseModel):
    """Schema for a credit prediction response."""
    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted default probability")
    risk_band: RiskBand = Field(..., description="Risk classification band")


def classify_risk(probability: float) -> RiskBand:
    """Classify a default probability into a risk band."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict(application: CreditApplication) -> PredictionResult:
    """
    Run a credit default prediction for a single application.

    Parameters
    ----------
    application : CreditApplication
        Validated application features.

    Returns
    -------
    PredictionResult
        Prediction with approval decision, probability, and risk band.
    """
    mdl = model_module.get_model()
    scaler = model_module.get_scaler()
    feature_names = model_module.get_feature_names()

    # Build a full 18-feature vector matching the training schema.
    # Feature order from feature_names.pkl:
    # ['credit_score', 'annual_income', 'debt_to_income', 'employment_years',
    #  'loan_amount', 'interest_rate', 'verified_income', 'num_credit_lines',
    #  'delinquency_2yrs', 'loan_purpose_business', 'loan_purpose_debt_consolidation',
    #  'loan_purpose_home_improvement', 'loan_purpose_major_purchase',
    #  'loan_purpose_other', 'home_ownership_MORTGAGE', 'home_ownership_OTHER',
    #  'home_ownership_OWN', 'home_ownership_RENT']
    home_upper = application.home_ownership.upper()

    raw_features = [
        application.credit_score,       # credit_score
        application.income,              # annual_income
        application.debt_to_income,      # debt_to_income
        application.employment_years,    # employment_years
        0.0,                            # loan_amount (not in API schema — default)
        0.0,                            # interest_rate (not in API schema — default)
        application.verified_income,     # verified_income
        application.loan_history_count, # num_credit_lines
        0,                              # delinquency_2yrs (no data in API schema)
        0,                              # loan_purpose_business
        0,                              # loan_purpose_debt_consolidation
        0,                              # loan_purpose_home_improvement
        0,                              # loan_purpose_major_purchase
        1 if application.home_ownership == "other" else 0,  # loan_purpose_other
        1 if home_upper == "MORTGAGE" else 0,  # home_ownership_MORTGAGE
        1 if application.home_ownership == "other" else 0,   # home_ownership_OTHER
        1 if home_upper == "OWN" else 0,       # home_ownership_OWN
        1 if home_upper == "RENT" else 0,      # home_ownership_RENT
    ]

    feature_vector = scaler.transform([raw_features])

    probability = float(mdl.predict_proba(feature_vector)[0, 1])

    risk_band = classify_risk(probability)
    approved = probability < 0.35

    return PredictionResult(
        approved=approved,
        default_probability=round(probability, 4),
        risk_band=risk_band,
    )