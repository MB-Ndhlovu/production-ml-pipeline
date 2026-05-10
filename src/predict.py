"""Prediction schemas and functions."""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd

from src.model import get_model


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


HOME_OWNERSHIP_MAP = {"rent": "RENT", "own": "OWN", "mortgage": "MORTGAGE", "other": "OTHER"}
LOAN_PURPOSE_KEYS = [
    "loan_purpose_business",
    "loan_purpose_debt_consolidation",
    "loan_purpose_home_improvement",
    "loan_purpose_major_purchase",
    "loan_purpose_other",
]
HOME_OWNERSHIP_KEYS = [
    "home_ownership_MORTGAGE",
    "home_ownership_OTHER",
    "home_ownership_OWN",
    "home_ownership_RENT",
]


class CreditApplication(BaseModel):
    """Input schema for credit scoring predictions."""
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: int = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., gt=0, description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        default="rent", description="Home ownership status"
    )
    verified_income: int = Field(default=0, ge=0, le=1, description="Income verified (0 or 1)")
    loan_amount: Optional[float] = Field(default=10000.0, gt=0, description="Requested loan amount")
    interest_rate: Optional[float] = Field(default=10.0, gt=0, le=30, description="Interest rate offered")
    num_credit_lines: Optional[int] = Field(default=3, ge=0, description="Number of open credit lines")
    delinquency_2yrs: Optional[int] = Field(default=0, ge=0, description="Delinquencies in past 2 years")
    loan_purpose: Optional[Literal["business", "debt_consolidation", "home_improvement",
                                   "major_purchase", "other"]] = Field(
        default="other", description="Purpose of the loan"
    )


def get_risk_band(probability: float) -> RiskBand:
    """Determine risk band from default probability."""
    if probability < 0.15:
        return RiskBand.LOW
    elif probability <= 0.35:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH


def predict(application: CreditApplication) -> dict:
    """
    Make a prediction for the given credit application.

    Returns a dict with approved status, default probability, and risk band.
    """
    model, scaler, feature_names = get_model()

    raw = {
        "credit_score": application.credit_score,
        "annual_income": application.income,
        "debt_to_income": application.debt_to_income,
        "employment_years": application.employment_years,
        "loan_amount": application.loan_amount,
        "interest_rate": application.interest_rate,
        "verified_income": application.verified_income,
        "num_credit_lines": application.num_credit_lines,
        "delinquency_2yrs": application.delinquency_2yrs,
    }
    for key in LOAN_PURPOSE_KEYS:
        raw[key] = 1 if application.loan_purpose in key.replace("loan_purpose_", "") else 0
    for key in HOME_OWNERSHIP_KEYS:
        raw[key] = 0
    home_key = f"home_ownership_{HOME_OWNERSHIP_MAP.get(application.home_ownership, 'RENT')}"
    raw[home_key] = 1

    df = pd.DataFrame([[raw[name] for name in feature_names]], columns=feature_names)
    scaled = scaler.transform(df)

    prob = model.predict_proba(scaled)[0][1]
    approved = prob < 0.35

    return {
        "approved": bool(approved),
        "default_probability": round(float(prob), 4),
        "risk_band": get_risk_band(prob),
    }