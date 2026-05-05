from pydantic import BaseModel, Field
from typing import Literal


class PredictionInput(BaseModel):
    """Schema for credit risk prediction input."""

    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score")
    employment_years: int = Field(..., description="Years of employment")
    debt_to_income: float = Field(..., description="Debt to income ratio")
    loan_history_count: int = Field(..., description="Number of past loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        default="rent", description="Home ownership status"
    )
    verified_income: int = Field(..., description="Income verified (1=yes, 0=no)")

    class Config:
        json_schema_extra = {
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


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict_default(input_data: PredictionInput) -> dict:
    """Run credit risk prediction on input features.

    Returns a dict with approved (bool), default_probability (float),
    and risk_band (Literal['low','medium','high']).
    """
    from src.model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    raw_features = {
        "income": input_data.income,
        "credit_score": input_data.credit_score,
        "employment_years": input_data.employment_years,
        "debt_to_income": input_data.debt_to_income,
        "loan_history_count": input_data.loan_history_count,
        "age": input_data.age,
        f"home_ownership_{input_data.home_ownership}": 1,
        "verified_income": input_data.verified_income,
    }

    feature_vector = [raw_features.get(fn, 0) for fn in feature_names]
    scaled = scaler.transform([feature_vector])

    prob = float(model.predict_proba(scaled)[0][1])
    approved = prob < 0.35

    return {
        "approved": approved,
        "default_probability": round(prob, 4),
        "risk_band": get_risk_band(prob),
    }