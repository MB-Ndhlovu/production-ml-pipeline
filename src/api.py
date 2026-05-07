"""FastAPI application with /health and /predict endpoints."""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.model import is_model_loaded
from src.predict import CreditApplication, PredictionResponse, predict_default

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction.",
    version="1.0.0",
)


@app.get("/health", response_model=dict, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns the server status and whether the model artifacts are loaded.
    """
    loaded = is_model_loaded()
    return JSONResponse({"status": "ok", "model_loaded": loaded})


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
async def predict(application: CreditApplication):
    """
    Score a credit application.

    Accepts applicant features, runs the model, and returns the default
    probability, approval decision, and risk band.
    """
    return predict_default(application)