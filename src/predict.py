"""Prediction logic and Pydantic schemas."""
from enum import Enum
from pydantic import BaseModel, Field

from src.model import load_model, load_scaler, load_feature_names


class RiskBand(str, Enum):
    """Risk classification bands."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CreditApplication(BaseModel):
    """Input schema for a credit application."""
    income: float = Field(..., description="Annual income in ZAR")
    credit_score: int = Field(..., description="Credit score (300–850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio")
    loan_history_count: int = Field(..., ge=0, description="Number of prior loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: str = Field(..., description="'rent', 'own', or 'mortgage'")
    verified_income: int = Field(..., description="Income verified (0 or 1)")
    interest_rate: float = Field(
        default=0.12,
        description="Interest rate as decimal fraction (e.g., 0.12 for 12%)"
    )


class PredictionResponse(BaseModel):
    """API response for a prediction."""
    approved: bool
    default_probability: float
    risk_band: RiskBand


def classify_risk(probability: float) -> RiskBand:
    """Classify probability into a risk band."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict_default(application: CreditApplication) -> PredictionResponse:
    """
    Run the credit model on a single application.

    Returns the default probability, approval decision, and risk band.
    Approval is granted when probability < 0.25.
    """
    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()

    # Build feature vector — map user-friendly names to model features
    raw_features = {
        "annual_income": application.income,
        "credit_score": application.credit_score,
        "employment_years": application.employment_years,
        "debt_to_income": application.debt_to_income,
        "num_credit_lines": application.loan_history_count,
        "verified_income": application.verified_income,
        "interest_rate": application.interest_rate,
        # Defaults for features not exposed in the public API
        "loan_amount": 10000,
        "delinquency_2yrs": 0,
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 1,
    }

    # Home ownership one-hot (model uses uppercase prefix)
    home = application.home_ownership.lower()
    for val, prefix in [("rent", "RENT"), ("own", "OWN"), ("mortgage", "MORTGAGE")]:
        raw_features[f"home_ownership_{prefix}"] = int(home == val)
    raw_features["home_ownership_OTHER"] = 0

    # Assemble in the exact order the model expects
    feature_vector = [[raw_features[f] for f in feature_names]]

    # Scale and predict
    scaled = scaler.transform(feature_vector)
    probability = float(model.predict_proba(scaled)[0, 1])
    approved = probability < 0.25

    return PredictionResponse(
        approved=approved,
        default_probability=round(probability, 6),
        risk_band=classify_risk(probability),
    )