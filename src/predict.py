from enum import Enum
from pydantic import BaseModel, Field


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


class PredictionRequest(BaseModel):
    """Credit scoring prediction request."""

    income: float = Field(..., gt=0, description="Annual income in currency units")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=2, description="Debt-to-income ratio")
    loan_history_count: int = Field(..., ge=0, description="Number of loans in history")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0 or 1)")
    loan_purpose: LoanPurpose = Field(
        default=LoanPurpose.OTHER, description="Purpose of the loan"
    )
    loan_amount: float = Field(
        default=10000.0, description="Requested loan amount"
    )
    interest_rate: float = Field(
        default=0.15, description="Proposed interest rate"
    )
    num_credit_lines: int = Field(
        default=3, description="Number of active credit lines"
    )
    delinquency_2yrs: int = Field(
        default=0, description="Number of delinquencies in last 2 years"
    )

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


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionResponse(BaseModel):
    """Credit scoring prediction response."""

    approved: bool = Field(..., description="Whether the application is approved")
    default_probability: float = Field(..., ge=0, le=1, description="Predicted probability of default")
    risk_band: RiskBand = Field(..., description="Risk classification band")


def predict_default(request: PredictionRequest) -> PredictionResponse:
    """Run prediction on a credit application."""
    import pandas as pd
    from src.model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    row = {
        "credit_score": request.credit_score,
        "annual_income": request.income,
        "debt_to_income": request.debt_to_income,
        "employment_years": request.employment_years,
        "loan_amount": request.loan_amount,
        "interest_rate": request.interest_rate,
        "verified_income": request.verified_income,
        "num_credit_lines": request.num_credit_lines,
        "delinquency_2yrs": request.delinquency_2yrs,
        "loan_purpose_business": 1 if request.loan_purpose == LoanPurpose.BUSINESS else 0,
        "loan_purpose_debt_consolidation": 1 if request.loan_purpose == LoanPurpose.DEBT_CONSOLIDATION else 0,
        "loan_purpose_home_improvement": 1 if request.loan_purpose == LoanPurpose.HOME_IMPROVEMENT else 0,
        "loan_purpose_major_purchase": 1 if request.loan_purpose == LoanPurpose.MAJOR_PURCHASE else 0,
        "loan_purpose_other": 1 if request.loan_purpose == LoanPurpose.OTHER else 0,
        "home_ownership_MORTGAGE": 1 if request.home_ownership == HomeOwnership.MORTGAGE else 0,
        "home_ownership_OTHER": 1 if request.home_ownership == HomeOwnership.OTHER else 0,
        "home_ownership_OWN": 1 if request.home_ownership == HomeOwnership.OWN else 0,
        "home_ownership_RENT": 1 if request.home_ownership == HomeOwnership.RENT else 0,
    }

    X_df = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X_df)

    prob = float(model.predict_proba(X_scaled)[0, 1])

    if prob < 0.15:
        band = RiskBand.LOW
    elif prob <= 0.35:
        band = RiskBand.MEDIUM
    else:
        band = RiskBand.HIGH

    return PredictionResponse(approved=prob < 0.5, default_probability=round(prob, 4), risk_band=band)