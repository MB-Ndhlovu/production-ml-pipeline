"""FastAPI application for the credit scoring service."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .model import is_model_loaded
from .predict import PredictionInput, PredictionOutput, predict


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
)


@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint.

    Returns the service status and whether model artifacts are loaded.

    Returns:
        dict: {"status": "ok", "model_loaded": bool}
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": is_model_loaded(),
        }
    )


@app.post("/predict", response_model=PredictionOutput)
async def credit_predict(input_data: PredictionInput):
    """Credit scoring prediction endpoint.

    Accepts applicant features and returns approval decision,
    default probability, and risk classification.

    Args:
        input_data: PredictionInput with applicant features

    Returns:
        PredictionOutput with approved, default_probability, and risk_band
    """
    return predict(input_data)
