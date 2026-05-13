"""Pydantic schemas and prediction logic for the credit scoring API."""

from enum import Enum
from pydantic import BaseModel, Field

from src.model import load_model, load_scaler, load_feature_names


class HomeOwnership(str, Enum):
    OWN = "own"
    RENT = "rent"
    MORTGAGE = "mortgage"
    OTHER = "other"


class LoanPurpose(str, Enum):
    BUSINESS = "business"
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    OTHER = "other"


class PredictionInput(BaseModel):
    """Input schema for a single credit scoring prediction."""

    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of loans in history")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., description="Income verified (0 or 1)")
    loan_amount: float = Field(default=10000, gt=0, description="Requested loan amount")
    interest_rate: float = Field(default=0.10, ge=0, le=1, description="Interest rate")
    loan_purpose: LoanPurpose = Field(default=LoanPurpose.OTHER, description="Purpose of the loan")

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
                "loan_amount": 10000,
                "interest_rate": 0.10,
                "loan_purpose": "other"
            }
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for a credit scoring prediction."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Probability of default")
    risk_band: str = Field(..., description="Risk band: low, medium, or high")

    model_config = {
        "json_schema_extra": {
            "example": {
                "approved": True,
                "default_probability": 0.12,
                "risk_band": "low"
            }
        }
    }


# Load artifacts at module level
_model = load_model()
_scaler = load_scaler()
_feature_names = load_feature_names()


def predict(input_data: PredictionInput) -> PredictionOutput:
    """Make a credit scoring prediction for the given input data."""

    # Build feature vector matching the order in _feature_names
    loan_purpose_key = f"loan_purpose_{input_data.loan_purpose.value}"
    home_key = f"home_ownership_{input_data.home_ownership.value.upper()}"

    features = {
        "credit_score": float(input_data.credit_score),
        "annual_income": float(input_data.income),
        "debt_to_income": float(input_data.debt_to_income),
        "employment_years": float(input_data.employment_years),
        "loan_amount": float(input_data.loan_amount),
        "interest_rate": float(input_data.interest_rate),
        "verified_income": float(input_data.verified_income),
        "num_credit_lines": float(input_data.loan_history_count),
        "delinquency_2yrs": 0.0,
        "loan_purpose_business": 1.0 if loan_purpose_key == "loan_purpose_business" else 0.0,
        "loan_purpose_debt_consolidation": 1.0 if loan_purpose_key == "loan_purpose_debt_consolidation" else 0.0,
        "loan_purpose_home_improvement": 1.0 if loan_purpose_key == "loan_purpose_home_improvement" else 0.0,
        "loan_purpose_major_purchase": 1.0 if loan_purpose_key == "loan_purpose_major_purchase" else 0.0,
        "loan_purpose_other": 1.0 if loan_purpose_key == "loan_purpose_other" else 0.0,
        "home_ownership_MORTGAGE": 1.0 if home_key == "home_ownership_MORTGAGE" else 0.0,
        "home_ownership_OTHER": 1.0 if home_key == "home_ownership_OTHER" else 0.0,
        "home_ownership_OWN": 1.0 if home_key == "home_ownership_OWN" else 0.0,
        "home_ownership_RENT": 1.0 if home_key == "home_ownership_RENT" else 0.0,
    }

    ordered_features = [features[name] for name in _feature_names]

    import numpy as np
    X = np.array([ordered_features], dtype=float)
    X_scaled = _scaler.transform(X)

    default_prob = float(_model.predict_proba(X_scaled)[0][1])

    # Determine risk band
    if default_prob < 0.15:
        risk_band = "low"
    elif default_prob <= 0.35:
        risk_band = "medium"
    else:
        risk_band = "high"

    approved = default_prob < 0.35

    return PredictionOutput(
        approved=approved,
        default_probability=round(default_prob, 4),
        risk_band=risk_band
    )


def is_model_loaded() -> bool:
    """Check whether the model artifacts loaded successfully."""
    return _model is not None