"""FastAPI application for the credit scoring prediction API."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.predict import CreditApplication, PredictionResult, predict


_model_loaded = False


def _try_load_model():
    """Try to load model artifacts, return True on success."""
    try:
        from src.model import load_model, load_scaler, load_feature_names
        load_model()
        load_scaler()
        load_feature_names()
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attempt to load model artifacts on startup."""
    global _model_loaded
    _model_loaded = _try_load_model()
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=dict,
    tags=["health"],
    summary="Health check",
    responses={200: {"description": "Service is healthy"}},
)
def health():
    """
    Check API health and model loading status.

    Returns the service status and whether model artifacts
    were successfully loaded on startup.
    """
    return {"status": "ok", "model_loaded": _model_loaded}


@app.post(
    "/predict",
    response_model=PredictionResult,
    tags=["prediction"],
    summary="Predict credit default",
    responses={
        200: {"description": "Prediction returned successfully"},
        400: {"description": "Invalid input data"},
        422: {"description": "Validation error"},
    },
)
def predict_endpoint(application: CreditApplication):
    """
    Predict credit approval, default probability, and risk band.

    Accepts a credit application with income, credit score, employment
    history, and other features. Returns whether the application is
    approved, the probability of default, and a risk classification.
    """
    global _model_loaded
    if not _model_loaded:
        _model_loaded = _try_load_model()

    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return predict(application)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e