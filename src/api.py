"""FastAPI application for the credit scoring ML pipeline."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import is_model_loaded
from src.predict import PredictInput, PredictOutput, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate model loading on startup."""
    if not is_model_loaded():
        raise RuntimeError("Failed to load model artifacts on startup")
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for real-time credit default prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="Health check",
    response_model=dict,
    tags=["health"],
)
async def health():
    """
    Return API health status and whether the model is loaded.

    Returns 200 with status ok and model_loaded flag.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": is_model_loaded(),
        }
    )


@app.post(
    "/predict",
    summary="Predict credit default",
    response_model=PredictOutput,
    tags=["prediction"],
    responses={
        200: {"description": "Prediction returned successfully"},
        422: {"description": "Validation error in request body"},
        500: {"description": "Internal prediction error"},
    },
)
async def predict_endpoint(input_data: PredictInput):
    """
    Accept applicant features and return a credit default prediction.

    Returns:
        - approved: bool — loan approved if default_probability < 0.5
        - default_probability: float — predicted probability of default
        - risk_band: str — one of "low", "medium", "high"
    """
    try:
        result = predict(input_data)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc