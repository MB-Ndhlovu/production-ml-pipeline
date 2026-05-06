from typing import Literal
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input schema for credit default prediction."""

    income: float = Field(..., description="Annual income in currency units")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: int = Field(..., description="Years of continuous employment")
    debt_to_income: float = Field(..., description="Ratio of total debt to annual income")
    loan_history_count: int = Field(..., description="Number of past credit lines")
    age: int = Field(..., description="Applicant age in years")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        ..., description="Home ownership status"
    )
    verified_income: int = Field(..., description="Income verification status (0 or 1)")

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


class PredictionOutput(BaseModel):
    """Output schema for credit default prediction."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(
        ..., description="Predicted probability of default (0-1)"
    )
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk classification based on default probability"
    )


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Determine risk band based on default probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict_default(input_data: PredictionInput) -> PredictionOutput:
    """
    Make a credit default prediction.

    Args:
        input_data: Validated prediction input features.

    Returns:
        PredictionOutput with approval status, probability, and risk band.
    """
    from .model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    feature_map = {
        "credit_score": input_data.credit_score,
        "annual_income": input_data.income,
        "debt_to_income": input_data.debt_to_income,
        "employment_years": input_data.employment_years,
        "loan_amount": 0,
        "interest_rate": 0,
        "verified_income": input_data.verified_income,
        "num_credit_lines": input_data.loan_history_count,
        "delinquency_2yrs": 0,
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 0,
        "home_ownership_MORTGAGE": 1 if input_data.home_ownership == "mortgage" else 0,
        "home_ownership_OTHER": 1 if input_data.home_ownership == "other" else 0,
        "home_ownership_OWN": 1 if input_data.home_ownership == "own" else 0,
        "home_ownership_RENT": 1 if input_data.home_ownership == "rent" else 0,
    }

    features = [feature_map[name] for name in feature_names]
    features_scaled = scaler.transform([features])
    probability = model.predict_proba(features_scaled)[0][1]
    approved = probability < 0.35
    risk_band = get_risk_band(probability)

    return PredictionOutput(
        approved=approved, default_probability=round(probability, 4), risk_band=risk_band
    )