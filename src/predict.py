"""Prediction logic and Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel, Field
from src.model import get_model, get_scaler, get_feature_names


class PredictInput(BaseModel):
    """Input features for credit default prediction."""

    income: float = Field(..., description="Annual income in currency units")
    credit_score: int = Field(..., description="Credit score (300–850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio (0–1)")
    loan_history_count: int = Field(..., description="Number of past loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        default="other", description="Home ownership status"
    )
    verified_income: int = Field(default=0, description="Income verified (0/1)")

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
    """Prediction result."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., description="Predicted default probability")
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk band based on default probability"
    )


def _build_feature_vector(input_data: PredictInput) -> list:
    """Build a 18-element feature vector matching the training feature order."""
    # One-hot encoding for home_ownership
    home_ownership_upper = input_data.home_ownership.upper()
    home_ownership_MORTGAGE = 1 if home_ownership_upper == "MORTGAGE" else 0
    home_ownership_OTHER = 1 if home_ownership_upper == "OTHER" else 0
    home_ownership_OWN = 1 if home_ownership_upper == "OWN" else 0
    home_ownership_RENT = 1 if home_ownership_upper == "RENT" else 0

    # All loan purpose features default to 0 (not provided in input)
    loan_purpose_business = 0
    loan_purpose_debt_consolidation = 0
    loan_purpose_home_improvement = 0
    loan_purpose_major_purchase = 0
    loan_purpose_other = 1  # default

    return [
        input_data.credit_score,        # credit_score
        input_data.income,               # annual_income
        input_data.debt_to_income,       # debt_to_income
        input_data.employment_years,    # employment_years
        0,                               # loan_amount (not in input, use 0)
        0,                               # interest_rate (not in input, use 0)
        input_data.verified_income,     # verified_income
        input_data.loan_history_count,  # num_credit_lines
        0,                               # delinquency_2yrs (not in input)
        loan_purpose_business,
        loan_purpose_debt_consolidation,
        loan_purpose_home_improvement,
        loan_purpose_major_purchase,
        loan_purpose_other,
        home_ownership_MORTGAGE,
        home_ownership_OTHER,
        home_ownership_OWN,
        home_ownership_RENT,
    ]


def predict_default(input_data: PredictInput) -> PredictOutput:
    """
    Predict credit default probability for a given application.

    Args:
        input_data: Validated input features.

    Returns:
        PredictOutput with approval decision, probability, and risk band.
    """
    model = get_model()
    scaler = get_scaler()

    feature_vector = _build_feature_vector(input_data)
    scaled = scaler.transform([feature_vector])
    prob = float(model.predict_proba(scaled)[0][1])

    # Determine risk band
    if prob < 0.15:
        risk_band: Literal["low", "medium", "high"] = "low"
    elif prob <= 0.35:
        risk_band = "medium"
    else:
        risk_band = "high"

    approved = prob < 0.35

    return PredictOutput(
        approved=approved, default_probability=round(prob, 4), risk_band=risk_band
    )