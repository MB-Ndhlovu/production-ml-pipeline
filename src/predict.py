"""Prediction schemas and functions."""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from .model import get_model, get_scaler, get_feature_names


class PredictionInput(BaseModel):
    """Input schema for credit default prediction."""

    income: float = Field(..., gt=0, description="Annual income in currency units")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years of employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of previous loans")
    age: int = Field(..., ge=18, le=120, description="Age in years")
    home_ownership: Literal["rent", "own", "mortgage"] = Field(..., description="Home ownership status")
    verified_income: int = Field(..., ge=0, le=1, description="Income verification status (0 or 1)")

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
    """Output schema for credit default prediction."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted probability of default")
    risk_band: Literal["low", "medium", "high"] = Field(..., description="Risk classification band")

    model_config = {
        "json_schema_extra": {
            "example": {
                "approved": True,
                "default_probability": 0.12,
                "risk_band": "low"
            }
        }
    }


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Determine risk band from probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict_default(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict credit default probability.

    Args:
        input_data: Validated prediction input

    Returns:
        Prediction output with approval decision, probability, and risk band
    """
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    import pandas as pd

    # Build feature vector matching the exact feature names the model was trained on
    features = {
        "credit_score": input_data.credit_score,
        "annual_income": input_data.income,
        "debt_to_income": input_data.debt_to_income,
        "employment_years": input_data.employment_years,
        "loan_amount": 10000,
        "interest_rate": 0.15,
        "verified_income": input_data.verified_income,
        "num_credit_lines": input_data.loan_history_count,
        "delinquency_2yrs": 0,
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 1,
        "home_ownership_MORTGAGE": 0,
        "home_ownership_OTHER": 0,
        "home_ownership_OWN": 0,
        "home_ownership_RENT": 0
    }

    home_ownership_map = {
        "rent": "home_ownership_RENT",
        "own": "home_ownership_OWN",
        "mortgage": "home_ownership_MORTGAGE"
    }
    features[home_ownership_map.get(input_data.home_ownership, "home_ownership_RENT")] = 1

    df = pd.DataFrame([features], columns=feature_names)

    scaled = scaler.transform(df)
    probability = model.predict_proba(scaled)[0][1]

    approved = probability <= 0.35

    return PredictionOutput(
        approved=approved,
        default_probability=round(float(probability), 4),
        risk_band=get_risk_band(probability)
    )