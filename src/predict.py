from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from src.model import get_model, get_scaler, get_feature_names


class HomeOwnership(str, Enum):
    """Home ownership status."""
    rent = "rent"
    own = "own"
    mortgage = "mortgage"
    other = "other"


class PredictionInput(BaseModel):
    """Input schema for credit default prediction."""
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt to income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of previous loans")
    age: int = Field(..., ge=18, le=120, description="Age in years")
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


RiskBand = Literal["low", "medium", "high"]


def get_risk_band(probability: float) -> RiskBand:
    """Classify probability into risk band."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def _build_feature_vector(input_data: PredictionInput) -> pd.DataFrame:
    """
    Build a feature DataFrame matching the 18 features the scaler expects.

    Feature names: credit_score, annual_income, debt_to_income, employment_years,
    loan_amount, interest_rate, verified_income, num_credit_lines, delinquency_2yrs,
    loan_purpose_business, loan_purpose_debt_consolidation, loan_purpose_home_improvement,
    loan_purpose_major_purchase, loan_purpose_other, home_ownership_MORTGAGE,
    home_ownership_OTHER, home_ownership_OWN, home_ownership_RENT
    """
    ho = input_data.home_ownership.value.upper()

    default_loan_amount = 15000.0
    default_interest_rate = 12.0

    features = pd.DataFrame({
        "credit_score": [input_data.credit_score],
        "annual_income": [input_data.income],
        "debt_to_income": [input_data.debt_to_income],
        "employment_years": [input_data.employment_years],
        "loan_amount": [default_loan_amount],
        "interest_rate": [default_interest_rate],
        "verified_income": [input_data.verified_income],
        "num_credit_lines": [input_data.loan_history_count],
        "delinquency_2yrs": [0],
        "loan_purpose_business": [0],
        "loan_purpose_debt_consolidation": [1],
        "loan_purpose_home_improvement": [0],
        "loan_purpose_major_purchase": [0],
        "loan_purpose_other": [0],
        "home_ownership_MORTGAGE": [1 if ho == "MORTGAGE" else 0],
        "home_ownership_OTHER": [1 if ho == "OTHER" else 0],
        "home_ownership_OWN": [1 if ho == "OWN" else 0],
        "home_ownership_RENT": [1 if ho == "RENT" else 0],
    })

    return features


def predict_default(input_data: PredictionInput) -> tuple[float, RiskBand]:
    """
    Predict default probability for given input features.

    Args:
        input_data: Validated prediction input

    Returns:
        Tuple of (default_probability, risk_band)
    """
    model = get_model()
    scaler = get_scaler()

    features_df = _build_feature_vector(input_data)

    features_scaled = scaler.transform(features_df)

    probability = model.predict_proba(features_scaled)[0][1]

    risk_band = get_risk_band(probability)

    return probability, risk_band