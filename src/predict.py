from pydantic import BaseModel, Field
from typing import Literal


class PredictionInput(BaseModel):
    income: float = Field(..., description="Annual income")
    credit_score: int = Field(..., description="Credit score (300-850)")
    employment_years: int = Field(..., description="Years employed")
    debt_to_income: float = Field(..., description="Debt-to-income ratio")
    loan_history_count: int = Field(..., description="Number of past loans")
    age: int = Field(..., description="Applicant age")
    home_ownership: Literal["rent", "own", "mortgage"] = Field(..., description="Home ownership status")
    verified_income: int = Field(..., description="Income verified (0 or 1)")


class PredictionOutput(BaseModel):
    approved: bool
    default_probability: float
    risk_band: Literal["low", "medium", "high"]


def get_risk_band(prob: float) -> Literal["low", "medium", "high"]:
    if prob < 0.15:
        return "low"
    elif prob <= 0.35:
        return "medium"
    else:
        return "high"


def predict(input_data: PredictionInput, model, scaler, feature_names) -> PredictionOutput:
    import numpy as np

    # Build feature vector in correct order
    features = {
        "income": input_data.income,
        "credit_score": input_data.credit_score,
        "employment_years": input_data.employment_years,
        "debt_to_income": input_data.debt_to_income,
        "loan_history_count": input_data.loan_history_count,
        "age": input_data.age,
        "home_ownership": input_data.home_ownership,
        "verified_income": input_data.verified_income,
    }

    # Encode home_ownership
    for val in ["rent", "own", "mortgage"]:
        features[f"home_ownership_{val}"] = 1 if input_data.home_ownership == val else 0

    # Add missing columns with 0
    for name in feature_names:
        if name not in features:
            features[name] = 0

    X = np.array([[features[name] for name in feature_names]])
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0][1])
    approved = prob < 0.35
    band = get_risk_band(prob)

    return PredictionOutput(approved=approved, default_probability=round(prob, 4), risk_band=band)