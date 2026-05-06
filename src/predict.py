"""Prediction logic and Pydantic schemas for credit scoring."""

from pydantic import BaseModel, Field
from typing import Literal
import numpy as np


class PredictionInput(BaseModel):
    """Input schema for credit scoring prediction."""

    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years of employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of previous loans")
    age: int = Field(..., ge=18, le=120, description="Age in years")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        ..., description="Home ownership status"
    )
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0 or 1)")


class PredictionOutput(BaseModel):
    """Output schema for credit scoring prediction."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of default"
    )
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk classification band"
    )


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Determine risk band based on default probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict_default_probability(input_data: PredictionInput) -> float:
    """
    Predict the probability of credit default.

    Args:
        input_data: Validated prediction input features

    Returns:
        Default probability between 0 and 1
    """
    from src.model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    ho_upper = input_data.home_ownership.upper()
    ho_rent = 1 if ho_upper == "RENT" else 0
    ho_other = 1 if ho_upper == "OTHER" else 0
    ho_own = 1 if ho_upper == "OWN" else 0
    ho_mortgage = 1 if ho_upper == "MORTGAGE" else 0

    features_dict = {
        "credit_score": input_data.credit_score,
        "annual_income": input_data.income,
        "debt_to_income": input_data.debt_to_income,
        "employment_years": input_data.employment_years,
        "loan_amount": 10000,
        "interest_rate": 10.0,
        "verified_income": input_data.verified_income,
        "num_credit_lines": input_data.loan_history_count,
        "delinquency_2yrs": 0,
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 1,
        "home_ownership_MORTGAGE": ho_mortgage,
        "home_ownership_OTHER": ho_other,
        "home_ownership_OWN": ho_own,
        "home_ownership_RENT": ho_rent,
    }

    features = [features_dict[name] for name in feature_names]
    X = scaler.transform([features])
    probability = model.predict_proba(X)[0][1]

    return probability