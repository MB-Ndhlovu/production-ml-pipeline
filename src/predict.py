from pydantic import BaseModel, Field
from typing import Literal


class PredictionInput(BaseModel):
    """Input schema for credit default prediction."""

    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years of employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of previous loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
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


class PredictionOutput(BaseModel):
    """Output schema for credit default prediction."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of default"
    )
    risk_band: Literal["low", "medium", "high"] = Field(
        ..., description="Risk band based on probability"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "approved": True,
                "default_probability": 0.12,
                "risk_band": "low",
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


def is_approved(risk_band: Literal["low", "medium", "high"]) -> bool:
    """Determine approval from risk band."""
    return risk_band in ("low", "medium")