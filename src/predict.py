"""Prediction logic and Pydantic schemas."""

from pydantic import BaseModel, Field
from typing import Literal
from .model import credit_model


class PredictInput(BaseModel):
    """Schema for prediction request."""

    income: float = Field(..., description="Annual income")
    credit_score: float = Field(..., description="Credit score")
    employment_years: float = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio")
    loan_history_count: float = Field(..., description="Number of past loans")
    age: float = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        ..., description="Home ownership status"
    )
    verified_income: int = Field(..., ge=0, le=1, description="Income verified flag")

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


class PredictOutput(BaseModel):
    """Schema for prediction response."""

    approved: bool
    default_probability: float
    risk_band: Literal["low", "medium", "high"]


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Map probability to risk band."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict(input_data: PredictInput) -> PredictOutput:
    """
    Run a prediction through the credit model.

    Constructs the 18-feature vector matching the training feature order,
    applies scaling, runs inference, and returns approval + risk band.

    Returns PredictOutput with approved flag, probability, and risk band.
    """
    # Build 18-element vector matching feature_names order
    feature_vector = [
        input_data.credit_score,       # credit_score
        input_data.income,              # annual_income
        input_data.debt_to_income,      # debt_to_income
        input_data.employment_years,    # employment_years
        0.0,                            # loan_amount (not in schema, default 0)
        0.0,                            # interest_rate (not in schema, default 0)
        float(input_data.verified_income),  # verified_income
        input_data.loan_history_count,  # num_credit_lines
        0.0,                            # delinquency_2yrs (not in schema, default 0)
        0.0,                            # loan_purpose_business
        0.0,                            # loan_purpose_debt_consolidation
        0.0,                            # loan_purpose_home_improvement
        0.0,                            # loan_purpose_major_purchase
        1.0,                            # loan_purpose_other (default)
        1.0 if input_data.home_ownership == "mortgage" else 0.0,  # home_ownership_MORTGAGE
        1.0 if input_data.home_ownership == "other" else 0.0,    # home_ownership_OTHER
        1.0 if input_data.home_ownership == "own" else 0.0,     # home_ownership_OWN
        1.0 if input_data.home_ownership == "rent" else 0.0,     # home_ownership_RENT
    ]

    scaled = credit_model.scaler.transform([feature_vector])
    probability = credit_model.model.predict_proba(scaled)[0][1]

    approved = probability < 0.35
    risk_band = get_risk_band(probability)

    return PredictOutput(
        approved=approved,
        default_probability=round(float(probability), 4),
        risk_band=risk_band,
    )
