"""FastAPI application with /health and /predict endpoints."""
from fastapi import FastAPI
from src.predict import PredictInput, PredictOutput, predict_default
from src.model import is_model_loaded, load_artifacts

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction.",
    version="1.0.0",
)

# Load artifacts on startup
load_artifacts()


@app.get("/health", response_model=dict, tags=["health"])
def health_check() -> dict:
    """
    Health check endpoint.

    Returns the current status of the service and whether the model is loaded.
    """
    return {"status": "ok", "model_loaded": is_model_loaded()}


@app.post("/predict", response_model=PredictOutput, tags=["prediction"])
def predict(input_data: PredictInput) -> PredictOutput:
    """
    Predict credit default probability.

    Returns the approval decision, default probability, and risk band.
    Risk bands: low (prob < 0.15), medium (0.15–0.35), high (prob > 0.35).
    Approval threshold: probability < 0.35.
    """
    return predict_default(input_data)