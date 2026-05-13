"""FastAPI application exposing credit scoring endpoints."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.predict import PredictionInput, PredictionOutput, predict, is_model_loaded


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit scoring predictions.",
    version="1.0.0",
)


@app.get("/health", response_model=dict, tags=["Health"])
def health():
    """Health check endpoint.

    Returns the service status and whether the model is loaded.
    """
    return JSONResponse({
        "status": "ok",
        "model_loaded": is_model_loaded()
    })


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_endpoint(input_data: PredictionInput) -> PredictionOutput:
    """Credit scoring prediction endpoint.

    Accepts applicant features and returns a credit decision including
    default probability and risk band.
    """
    return predict(input_data)