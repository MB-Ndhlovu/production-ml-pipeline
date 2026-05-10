"""FastAPI application for credit scoring predictions."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from src.model import load_model, load_scaler, load_feature_names
from src.predict import (
    CreditApplication,
    PredictionResponse,
    get_risk_band,
    predict_default_probability,
)


model = None
scaler = None
feature_names = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global model, scaler, feature_names, model_loaded
    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()
    model_loaded = True
    yield
    model_loaded = False


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for real-time credit default predictions",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model_exclude_none=True)
async def health_check():
    """
    Health check endpoint.

    Returns the API status and whether the model artifacts are loaded.
    """
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    """
    Score a credit application.

    Accepts applicant features and returns the probability of default,
    along with an approval decision and risk band.
    """
    prob = predict_default_probability(application, model, scaler, feature_names)
    risk_band = get_risk_band(prob)
    approved = prob < 0.35

    return PredictionResponse(
        approved=approved,
        default_probability=round(prob, 4),
        risk_band=risk_band,
    )