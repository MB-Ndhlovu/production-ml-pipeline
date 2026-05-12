"""FastAPI application for the credit scoring prediction API."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import is_model_loaded
from src.predict import PredictInput, PredictOutput, predict_default


app = FastAPI(
    title="Credit Default Prediction API",
    description="Production ML pipeline for credit default probability estimation.",
    version="1.0.0",
)


@app.get(
    "/health",
    summary="Health check",
    response_model=dict,
    responses={200: {"description": "API is healthy"}},
)
def health():
    """
    Health check endpoint.

    Returns the current status of the API and whether the model artifacts
    have been successfully loaded.
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
    responses={200: {"description": "Prediction result"}},
)
def predict(input_data: PredictInput):
    """
    Run a credit default prediction.

    Accepts a JSON payload with application features and returns the default
    probability, risk band, and an approval decision.

    Risk bands:
    - **low**: probability < 0.15
    - **medium**: 0.15 ≤ probability ≤ 0.35
    - **high**: probability > 0.35
    """
    return predict_default(input_data)