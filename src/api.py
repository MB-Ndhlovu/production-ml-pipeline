from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np

from src.model import load_artifacts, get_model, get_scaler, get_feature_names, is_model_loaded
from src.predict import PredictionInput, PredictionOutput, get_risk_band, is_approved


# Mapping from our simplified input to the model's 18-feature schema
def build_feature_vector(input_data: PredictionInput) -> np.ndarray:
    """
    Build the 18-feature vector expected by the model.

    The model was trained with these features:
    ['credit_score', 'annual_income', 'debt_to_income', 'employment_years',
     'loan_amount', 'interest_rate', 'verified_income', 'num_credit_lines',
     'delinquency_2yrs', 'loan_purpose_business', 'loan_purpose_debt_consolidation',
     'loan_purpose_home_improvement', 'loan_purpose_major_purchase', 'loan_purpose_other',
     'home_ownership_MORTGAGE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT']
    """
    feature_names = get_feature_names()

    # Base features from input
    credit_score = input_data.credit_score
    annual_income = input_data.income
    debt_to_income = input_data.debt_to_income
    employment_years = input_data.employment_years
    verified_income = input_data.verified_income

    # Derived/count features - use defaults for missing fields
    num_credit_lines = input_data.loan_history_count
    delinquency_2yrs = 0  # Not in input, default to 0

    # Loan-specific features not in our input - use reasonable defaults
    loan_amount = 10000  # Default loan amount
    interest_rate = 0.12  # Default interest rate (as decimal, e.g. 0.12 = 12%)

    # One-hot encode home_ownership
    ho_mortgage = 1 if input_data.home_ownership == "mortgage" else 0
    ho_other = 1 if input_data.home_ownership == "other" else 0
    ho_own = 1 if input_data.home_ownership == "own" else 0
    ho_rent = 1 if input_data.home_ownership == "rent" else 0

    # Loan purpose - default to "other" category
    loan_purpose_business = 0
    loan_purpose_debt_consolidation = 0
    loan_purpose_home_improvement = 0
    loan_purpose_major_purchase = 0
    loan_purpose_other = 1

    # Build vector in the order expected by feature_names
    vector = [
        credit_score, annual_income, debt_to_income, employment_years,
        loan_amount, interest_rate, verified_income, num_credit_lines,
        delinquency_2yrs, loan_purpose_business, loan_purpose_debt_consolidation,
        loan_purpose_home_improvement, loan_purpose_major_purchase, loan_purpose_other,
        ho_mortgage, ho_other, ho_own, ho_rent
    ]

    return np.array(vector).reshape(1, -1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Credit Default Prediction API",
    description="Production ML pipeline for credit default risk prediction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=dict, tags=["Health"])
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns the API status and whether the model is loaded.
    """
    return {"status": "ok", "model_loaded": is_model_loaded()}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict credit default risk.

    Accepts applicant features and returns default probability with risk band.

    Risk bands:
    - low: probability < 0.15
    - medium: probability 0.15 - 0.35
    - high: probability > 0.35

    Approval is granted for low and medium risk bands.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()

    features = build_feature_vector(input_data)

    features_scaled = scaler.transform(features)
    prob = float(model.predict_proba(features_scaled)[0][1])

    risk_band = get_risk_band(prob)
    approved = is_approved(risk_band)

    return PredictionOutput(
        approved=approved,
        default_probability=round(prob, 4),
        risk_band=risk_band,
    )