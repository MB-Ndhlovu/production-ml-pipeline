"""Pydantic schemas and prediction logic."""
from enum import Enum

from pydantic import BaseModel, Field

from src.model import get_model, get_scaler, get_feature_names


class HomeOwnership(str, Enum):
    rent = "rent"
    own = "own"
    mortgage = "mortgage"
    other = "other"


class PredictRequest(BaseModel):
    """Input features for a single credit prediction."""

    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: float = Field(..., description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., ge=18, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., ge=0, le=1, description="1 if income verified, 0 otherwise")

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


class PredictResponse(BaseModel):
    """Prediction result."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., description="Predicted probability of default")
    risk_band: str = Field(..., description="Risk band: low, medium, or high")

    model_config = {
        "json_schema_extra": {
            "example": {
                "approved": True,
                "default_probability": 0.12,
                "risk_band": "low",
            }
        }
    }


def get_risk_band(probability: float) -> str:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict(request: PredictRequest) -> PredictResponse:
    """
    Run a credit default prediction from a validated request.

    Returns a ``PredictResponse`` with approval flag, probability,
    and risk band.
    """
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    # Build feature vector in the exact order expected by the model.
    # Defaults are used for features not exposed in the API.
    raw_features = {
        "credit_score": request.credit_score,
        "annual_income": request.income,
        "debt_to_income": request.debt_to_income,
        "employment_years": request.employment_years,
        "loan_amount": 10000.0,          # default
        "interest_rate": 0.15,            # default 15%
        "verified_income": request.verified_income,
        "num_credit_lines": request.loan_history_count,
        "delinquency_2yrs": 0,            # default
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 1,          # default to 'other'
        "home_ownership_MORTGAGE": 0,
        "home_ownership_OTHER": 0,
        "home_ownership_OWN": 0,
        "home_ownership_RENT": 0,
    }

    ho = request.home_ownership.value.upper()
    raw_features[f"home_ownership_{ho}"] = 1

    feature_vector = [raw_features[name] for name in feature_names]

    X = scaler.transform([feature_vector])
    probability = float(model.predict_proba(X)[0, 1])
    approved = probability < 0.35

    return PredictResponse(
        approved=approved,
        default_probability=round(probability, 4),
        risk_band=get_risk_band(probability),
    )