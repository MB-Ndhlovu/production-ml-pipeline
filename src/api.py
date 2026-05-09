from fastapi import FastAPI
from pydantic import BaseModel

from src.model import is_model_loaded, load_artifacts
from src.predict import predict, PredictionInput

# Load artifacts at startup
load_artifacts()

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction.",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Check API health and whether the model is loaded."""
    return HealthResponse(status="ok", model_loaded=is_model_loaded())


@app.post("/predict", tags=["predict"])
def predict_endpoint(input_data: PredictionInput) -> dict:
    """
    Make a credit default prediction.

    Returns approval decision, default probability, and risk band.
    """
    return predict(input_data)