from typing import Literal
from pydantic import BaseModel, Field


class PredictInput(BaseModel):
    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio")
    loan_history_count: int = Field(..., description="Number of previous loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage"] = Field(..., description="Home ownership status")
    verified_income: int = Field(..., description="Income verified (1=yes, 0=no)")
    loan_amount: float = Field(default=10000, description="Loan amount")
    interest_rate: float = Field(default=0.10, description="Interest rate")
    num_credit_lines: int = Field(default=2, description="Number of credit lines")
    delinquency_2yrs: int = Field(default=0, description="Delinquencies in last 2 years")
    loan_purpose: Literal["business", "debt_consolidation", "home_improvement", "major_purchase", "other"] = Field(
        default="other", description="Loan purpose"
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
                "loan_amount": 10000,
                "interest_rate": 0.10,
                "num_credit_lines": 2,
                "delinquency_2yrs": 0,
                "loan_purpose": "other"
            }
        }
    }


class PredictOutput(BaseModel):
    approved: bool = Field(..., description="Whether the loan is approved")
    default_probability: float = Field(..., description="Predicted probability of default")
    risk_band: Literal["low", "medium", "high"] = Field(..., description="Risk classification")


def get_risk_band(probability: float) -> Literal["low", "medium", "high"]:
    if probability < 0.15:
        return "low"
    elif probability <= 0.35:
        return "medium"
    else:
        return "high"


def predict(input_data: PredictInput) -> PredictOutput:
    import numpy as np
    from .model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    # Build feature dict with defaults for optional fields
    ho_upper = input_data.home_ownership.upper()
    lp = input_data.loan_purpose

    raw_features = {
        "credit_score": input_data.credit_score,
        "annual_income": input_data.income,
        "debt_to_income": input_data.debt_to_income,
        "employment_years": input_data.employment_years,
        "loan_amount": input_data.loan_amount,
        "interest_rate": input_data.interest_rate,
        "verified_income": input_data.verified_income,
        "num_credit_lines": input_data.num_credit_lines,
        "delinquency_2yrs": input_data.delinquency_2yrs,
        "loan_purpose_business": 1 if lp == "business" else 0,
        "loan_purpose_debt_consolidation": 1 if lp == "debt_consolidation" else 0,
        "loan_purpose_home_improvement": 1 if lp == "home_improvement" else 0,
        "loan_purpose_major_purchase": 1 if lp == "major_purchase" else 0,
        "loan_purpose_other": 1 if lp == "other" else 0,
        "home_ownership_MORTGAGE": 1 if ho_upper == "MORTGAGE" else 0,
        "home_ownership_OTHER": 1 if input_data.home_ownership == "other" else 0,
        "home_ownership_OWN": 1 if ho_upper == "OWN" else 0,
        "home_ownership_RENT": 1 if ho_upper == "RENT" else 0,
    }

    ordered = np.array([[raw_features[f] for f in feature_names]])
    scaled = scaler.transform(ordered)
    prob = model.predict_proba(scaled)[0][1]

    approved = prob < 0.35
    risk_band = get_risk_band(prob)

    return PredictOutput(
        approved=approved,
        default_probability=round(float(prob), 4),
        risk_band=risk_band
    )