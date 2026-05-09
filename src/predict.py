"""Pydantic schemas and prediction logic."""

from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.model import load_credit_model, load_scaler, load_feature_names

# Load artifacts once at module level
_model = load_credit_model()
_scaler = load_scaler()
_feature_names: List[str] = load_feature_names()


class HomeOwnership(str, Enum):
    """Home ownership status options."""

    rent = "rent"
    own = "own"
    mortgage = "mortgage"
    other = "other"


class LoanPurpose(str, Enum):
    """Loan purpose options."""

    business = "business"
    debt_consolidation = "debt_consolidation"
    home_improvement = "home_improvement"
    major_purchase = "major_purchase"
    other = "other"


class PredictionInput(BaseModel):
    """Input features for a single credit prediction."""

    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300–850)")
    employment_years: float = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0–1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., gt=0, description="Applicant age")
    home_ownership: HomeOwnership = Field(..., description="Home ownership status")
    verified_income: int = Field(..., description="Income verified (0 or 1)")
    loan_amount: Optional[float] = Field(10000, description="Requested loan amount")
    interest_rate: Optional[float] = Field(12.0, description="Interest rate")
    num_credit_lines: Optional[int] = Field(2, description="Number of active credit lines")
    delinquency_2yrs: Optional[int] = Field(0, ge=0, description="Delinquencies in last 2 years")
    loan_purpose: Optional[LoanPurpose] = Field(LoanPurpose.other, description="Purpose of the loan")


class RiskBand(str, Enum):
    """Risk band classification."""

    low = "low"
    medium = "medium"
    high = "high"


class PredictionOutput(BaseModel):
    """Output from the /predict endpoint."""

    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., description="Predicted probability of default")
    risk_band: RiskBand = Field(..., description="Risk band: low / medium / high")


def _build_dataframe(input_data: PredictionInput) -> pd.DataFrame:
    """Map Pydantic input to the model's expected feature columns."""

    home_ownership_map = {
        "rent":  "RENT",
        "own":   "OWN",
        "mortgage":  "MORTGAGE",
        "other": "OTHER",
    }
    ho_key = home_ownership_map[input_data.home_ownership.value]

    loan_purpose_map = {
        "business": "business",
        "debt_consolidation": "debt_consolidation",
        "home_improvement": "home_improvement",
        "major_purchase": "major_purchase",
        "other": "other",
    }
    lp_key = loan_purpose_map.get(input_data.loan_purpose.value, "other") if input_data.loan_purpose else "other"

    raw = {
        "credit_score":        input_data.credit_score,
        "annual_income":       input_data.income,
        "debt_to_income":      input_data.debt_to_income,
        "employment_years":    input_data.employment_years,
        "loan_amount":         input_data.loan_amount,
        "interest_rate":       input_data.interest_rate,
        "verified_income":     input_data.verified_income,
        "num_credit_lines":    input_data.num_credit_lines,
        "delinquency_2yrs":    input_data.delinquency_2yrs,
        "loan_purpose_business":              1 if lp_key == "business" else 0,
        "loan_purpose_debt_consolidation":    1 if lp_key == "debt_consolidation" else 0,
        "loan_purpose_home_improvement":       1 if lp_key == "home_improvement" else 0,
        "loan_purpose_major_purchase":         1 if lp_key == "major_purchase" else 0,
        "loan_purpose_other":                  1 if lp_key == "other" else 0,
        "home_ownership_MORTGAGE":            1 if ho_key == "MORTGAGE" else 0,
        "home_ownership_OTHER":                1 if ho_key == "OTHER" else 0,
        "home_ownership_OWN":                  1 if ho_key == "OWN" else 0,
        "home_ownership_RENT":                 1 if ho_key == "RENT" else 0,
    }

    df = pd.DataFrame([raw])
    return df[_feature_names]  # enforce correct column order


def predict(input_data: PredictionInput) -> PredictionOutput:
    """Run a credit prediction from raw input features.

    Returns a Pydantic output object with approval flag, default probability,
    and a risk band.
    """
    X = _build_dataframe(input_data)
    X_scaled = _scaler.transform(X)
    prob = float(_model.predict_proba(X_scaled)[0, 1])

    if prob < 0.15:
        band = RiskBand.low
        approved = True
    elif prob <= 0.35:
        band = RiskBand.medium
        approved = True
    else:
        band = RiskBand.high
        approved = False

    return PredictionOutput(
        approved=approved,
        default_probability=round(prob, 4),
        risk_band=band,
    )