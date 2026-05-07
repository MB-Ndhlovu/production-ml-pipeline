from typing import Literal

from pydantic import BaseModel, Field


class PredictInput(BaseModel):
    income: float = Field(..., gt=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300–850)")
    employment_years: float = Field(..., ge=0, description="Years employed")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0–1)")
    loan_history_count: int = Field(..., ge=0, description="Number of past loans")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage", "other"] = Field(
        ..., description="Home ownership status"
    )
    verified_income: int = Field(..., ge=0, le=1, description="Income verified (0 or 1)")

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


def predict(input_data: PredictInput) -> dict:
    """
    Run a prediction on a single applicant.

    Returns a dict with:
      - approved: bool
      - default_probability: float
      - risk_band: "low" | "medium" | "high"
    """
    from src.model import get_model, get_scaler, get_feature_names

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    raw_features = {
        "credit_score": input_data.credit_score,
        "annual_income": input_data.income,
        "debt_to_income": input_data.debt_to_income,
        "employment_years": input_data.employment_years,
        "loan_amount": input_data.income * 0.5,
        "interest_rate": 0.12,
        "verified_income": input_data.verified_income,
        "num_credit_lines": input_data.loan_history_count,
        "delinquency_2yrs": 0,
        "loan_purpose_business": 0,
        "loan_purpose_debt_consolidation": 1,
        "loan_purpose_home_improvement": 0,
        "loan_purpose_major_purchase": 0,
        "loan_purpose_other": 0,
        "home_ownership_MORTGAGE": 1 if input_data.home_ownership == "mortgage" else 0,
        "home_ownership_OTHER": 1 if input_data.home_ownership == "other" else 0,
        "home_ownership_OWN": 1 if input_data.home_ownership == "own" else 0,
        "home_ownership_RENT": 1 if input_data.home_ownership == "rent" else 0,
    }

    import pandas as pd

    X = pd.DataFrame([raw_features])[feature_names]
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0, 1])

    if prob < 0.15:
        band = "low"
        approved = True
    elif prob <= 0.35:
        band = "medium"
        approved = True
    else:
        band = "high"
        approved = False

    return {
        "approved": approved,
        "default_probability": round(prob, 4),
        "risk_band": band,
    }