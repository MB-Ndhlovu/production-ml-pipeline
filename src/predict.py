from enum import Enum
from pydantic import BaseModel, Field


class HomeOwnership(str, Enum):
    RENT = "rent"
    OWN = "own"
    MORTGAGE = "mortgage"
    OTHER = "other"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CreditApplication(BaseModel):
    """Input schema for credit scoring prediction."""
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of credit lines / loan history entries")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0 or 1)")


class PredictionResult(BaseModel):
    """Output schema for credit scoring prediction."""
    approved: bool
    default_probability: float
    risk_band: RiskBand


def get_risk_band(probability: float) -> RiskBand:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict(application: CreditApplication) -> PredictionResult:
    """Run prediction on a credit application."""
    from src.model import load_model, load_scaler, load_feature_names

    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()

    # Build feature vector in the correct order
    # Note: 'age' is accepted as input but is not used by this model
    raw = {
        "credit_score": application.credit_score,
        "annual_income": application.income,
        "debt_to_income": application.debt_to_income,
        "employment_years": application.employment_years,
        "loan_amount": 0,  # not provided in this API, use default
        "interest_rate": 0,  # not provided in this API, use default
        "verified_income": application.verified_income,
        "num_credit_lines": application.loan_history_count,
        "delinquency_2yrs": 0,  # not provided, use default
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 0,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 1,  # default
        "home_ownership_MORTGAGE": 0,
        "home_ownership_OTHER": 0,
        "home_ownership_OWN": 0,
        "home_ownership_RENT": 0,
    }

    ho = application.home_ownership.value.upper()
    if ho in raw:
        raw[ho] = 1

    # One-hot encode loan_purpose_other as default (already set above)

    import pandas as pd
    X = pd.DataFrame([[raw[f] for f in feature_names]], columns=feature_names)
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0, 1])
    approved = prob < 0.35  # approval threshold
    risk_band = get_risk_band(prob)

    return PredictionResult(
        approved=approved,
        default_probability=round(prob, 6),
        risk_band=risk_band,
    )
