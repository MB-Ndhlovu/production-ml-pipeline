"""Prediction logic and Pydantic schemas."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class RiskBand(str, Enum):
    """Risk classification bands based on default probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictInput(BaseModel):
    """Schema for a single credit application prediction request."""

    income: Annotated[float, Field(gt=0, description="Annual income in ZAR")]
    credit_score: Annotated[int, Field(ge=300, le=850, description="Credit score (300–850)")]
    employment_years: Annotated[int, Field(ge=0, description="Years in current employment")]
    debt_to_income: Annotated[float, Field(ge=0, le=1, description="Debt-to-income ratio (0–1)")]
    loan_history_count: Annotated[int, Field(ge=0, description="Number of past loans")]
    age: Annotated[int, Field(gt=18, description="Applicant age")]
    home_ownership: Annotated[str, Field(description="Own, rent, or mortgage")]
    verified_income: Annotated[int, Field(ge=0, le=1, description="Income verified (0/1)")]

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
    """Schema for a single credit application prediction response."""

    approved: bool
    default_probability: float
    risk_band: RiskBand


def _classify_band(probability: float) -> RiskBand:
    """Classify probability into a risk band."""
    if probability < 0.15:
        return RiskBand.LOW
    if probability <= 0.35:
        return RiskBand.MEDIUM
    return RiskBand.HIGH


def _one_hot(value: str, categories: list[str]) -> list[int]:
    """Return one-hot vector given the active category."""
    return [1 if value.lower() == cat.lower() else 0 for cat in categories]


def predict_default(input_data: PredictInput) -> PredictOutput:
    """
    Run a credit default prediction on a single application.

    Parameters
    ----------
    input_data : PredictInput
        Validated application features.

    Returns
    -------
    PredictOutput
        Prediction result with approval decision, probability, and risk band.
    """
    from src.model import get_feature_names, get_model, get_scaler

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    home_categories = ["MORTGAGE", "OTHER", "OWN", "RENT"]
    loan_purpose_categories = ["business", "debt_consolidation", "home_improvement", "major_purchase", "other"]

    home_oh = _one_hot(input_data.home_ownership, home_categories)
    purpose_oh = _one_hot("other", loan_purpose_categories)

    raw_features = [
        input_data.credit_score,       # credit_score
        input_data.income,              # annual_income
        input_data.debt_to_income,      # debt_to_income
        input_data.employment_years,    # employment_years
        10000.0,                        # loan_amount (default)
        0.15,                           # interest_rate (default)
        input_data.verified_income,     # verified_income
        input_data.loan_history_count,  # num_credit_lines
        0,                              # delinquency_2yrs (default)
        purpose_oh[0],                   # loan_purpose_business
        purpose_oh[1],                   # loan_purpose_debt_consolidation
        purpose_oh[2],                   # loan_purpose_home_improvement
        purpose_oh[3],                  # loan_purpose_major_purchase
        purpose_oh[4],                   # loan_purpose_other
        home_oh[0],                      # home_ownership_MORTGAGE
        home_oh[1],                      # home_ownership_OTHER
        home_oh[2],                      # home_ownership_OWN
        home_oh[3],                      # home_ownership_RENT
    ]

    assert len(raw_features) == len(feature_names), \
        f"Feature count mismatch: got {len(raw_features)}, expected {len(feature_names)}"

    scaled = scaler.transform([raw_features])
    probability = float(model.predict_proba(scaled)[0, 1])
    band = _classify_band(probability)
    approved = probability < 0.25

    return PredictOutput(
        approved=approved,
        default_probability=round(probability, 4),
        risk_band=band,
    )