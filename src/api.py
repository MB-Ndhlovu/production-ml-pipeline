from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import load_artifacts, is_model_loaded
from src.predict import PredictionInput, predict_default

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default predictions",
    version="1.0.0"
)

_model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load ML artifacts on startup."""
    global _model_loaded
    try:
        load_artifacts()
        _model_loaded = True
    except Exception as e:
        _model_loaded = False
        raise RuntimeError(f"Failed to load model: {e}")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    approved: bool
    default_probability: float
    risk_band: str


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and model are running correctly."
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Credit Default",
    description="Predict the probability of credit default and return risk classification."
)
async def predict(input_data: PredictionInput) -> PredictionResponse:
    """
    Predict credit default probability.

    Accepts feature data and returns the default probability along with
    a risk band classification (low, medium, high).
    """
    try:
        probability, risk_band = predict_default(input_data)

        approved = probability < 0.35

        return PredictionResponse(
            approved=approved,
            default_probability=round(probability, 4),
            risk_band=risk_band
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))