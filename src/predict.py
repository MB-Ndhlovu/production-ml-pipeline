"""Pydantic schemas and prediction logic."""

from enum import Enum
from pydantic import BaseModel, Field
import pandas as pd
from src.model import get_model, get_scaler, get_feature_names


class HomeOwnership(str, Enum):
    """Home ownership status."""
    OWN = "own"
    RENT = "rent"
    MORTGAGE = "mortgage"
    OTHER = "other"


class PredictionInput(BaseModel):
    """Input features for credit scoring prediction."""

    income: float = Field(..., gt=0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years of employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
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


class RiskBand(str, Enum):
    """Risk classification bands."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def get_risk_band(probability: float) -> RiskBand:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict_default(input_data: PredictionInput) -> dict:
    """
    Predict credit default probability and risk.

    Returns:
        dict with approved (bool), default_probability (float), and risk_band (str)
    """
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    # Build feature vector matching training features
    features = {
        "credit_score": input_data.credit_score,
        "annual_income": input_data.income,
        "debt_to_income": input_data.debt_to_income,
        "employment_years": input_data.employment_years,
        "loan_amount": input_data.income * 0.5,  # Estimate loan as 50% of income
        "interest_rate": 0.12,  # Default rate
        "verified_income": input_data.verified_income,
        "num_credit_lines": input_data.loan_history_count,
        "delinquency_2yrs": 0,  # Assume no recent delinquency
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 1,  # Default to other
        "home_ownership_MORTGAGE": 1 if input_data.home_ownership == "mortgage" else 0,
        "home_ownership_OTHER": 1 if input_data.home_ownership == "other" else 0,
        "home_ownership_OWN": 1 if input_data.home_ownership == "own" else 0,
        "home_ownership_RENT": 1 if input_data.home_ownership == "rent" else 0,
    }

    # Create DataFrame with correct column order
    X = pd.DataFrame([features])[feature_names]

    # Scale and predict
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0][1]

    risk_band = get_risk_band(proba)
    approved = proba < 0.35

    return {
        "approved": bool(approved),
        "default_probability": round(float(proba), 4),
        "risk_band": risk_band.value
    }