from typing import Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from src.model import get_model, get_scaler, get_feature_names


class PredictionInput(BaseModel):
    """Input schema for credit default prediction."""

    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: float = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., description="Number of past loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        ..., description="Home ownership status"
    )
    verified_income: int = Field(..., description="Income verified (0 or 1)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "income": 65000,
                    "credit_score": 720,
                    "employment_years": 5,
                    "debt_to_income": 0.28,
                    "loan_history_count": 2,
                    "age": 34,
                    "home_ownership": "rent",
                    "verified_income": 1,
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for credit default prediction."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(
        ..., description="Probability of default (0-1)"
    )
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk band based on probability"
    )


def _get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    return "high"


def _build_features(input_data: PredictionInput) -> pd.DataFrame:
    """Build feature DataFrame matching the trained model's expected schema."""
    scaler = get_scaler()
    feature_names_list = get_feature_names()

    home = input_data.home_ownership.upper()
    raw = {
        "annual_income": input_data.income,
        "credit_score": input_data.credit_score,
        "employment_years": input_data.employment_years,
        "debt_to_income": input_data.debt_to_income,
        "num_credit_lines": input_data.loan_history_count,
        "verified_income": input_data.verified_income,
        "age": input_data.age,
        "loan_amount": 0,
        "interest_rate": 0,
        "delinquency_2yrs": 0,
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 0,
        "home_ownership_MORTGAGE": 1 if home == "MORTGAGE" else 0,
        "home_ownership_OTHER": 1 if home == "OTHER" else 0,
        "home_ownership_OWN": 1 if home == "OWN" else 0,
        "home_ownership_RENT": 1 if home == "RENT" else 0,
    }

    df = pd.DataFrame([raw], dtype=np.float64)[feature_names_list]

    if scaler and hasattr(scaler, "transform"):
        df[:] = scaler.transform(df)

    return df


def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Make a credit default prediction.

    Args:
        input_data: Validated prediction input.

    Returns:
        PredictionOutput with approval decision, probability, and risk band.
    """
    features = _build_features(input_data)
    model = get_model()
    default_prob = float(model.predict_proba(features)[0][1])
    approved = default_prob < 0.35
    risk_band = _get_risk_band(default_prob)

    return PredictionOutput(
        approved=approved,
        default_probability=round(default_prob, 4),
        risk_band=risk_band,
    )