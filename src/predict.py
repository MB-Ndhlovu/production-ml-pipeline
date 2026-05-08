"""Pydantic schemas and prediction interface."""

from typing import Literal

from pydantic import BaseModel, Field

from src.model import predict as model_predict


class PredictInput(BaseModel):
    """Input schema for /predict endpoint."""

    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio")
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


class PredictOutput(BaseModel):
    """Output schema for /predict endpoint."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(
        ..., description="Estimated probability of default"
    )
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk classification band"
    )


def make_prediction(input_data: PredictInput) -> PredictOutput:
    """
    Run the credit scoring model on the given input.

    Args:
        input_data: Validated Pydantic input schema.

    Returns:
        PredictOutput with approval decision, probability, and risk band.
    """
    features = input_data.model_dump()
    prob, approved, risk_band = model_predict(features)
    return PredictOutput(
        approved=approved,
        default_probability=round(prob, 4),
        risk_band=risk_band,
    )