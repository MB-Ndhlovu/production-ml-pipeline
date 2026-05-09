"""FastAPI application for the credit scoring prediction service."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import model as model_module
from .predict import CreditApplication, PredictionResult, predict

app = FastAPI(
    title="Credit Scoring API",
    description="Real-time credit default prediction service.",
    version="1.0.0",
)


@app.get(
    "/health",
    summary="Health check",
    response_model=dict,
    responses={200: {"description": "Service is healthy"}},
)
def health():
    """
    Return the health status of the service and model loading state.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": model_module.is_model_loaded(),
        }
    )


@app.post(
    "/predict",
    summary="Predict credit default risk",
    response_model=PredictionResult,
    responses={
        200: {"description": "Prediction result"},
        400: {"description": "Invalid input"},
        500: {"description": "Model or server error"},
    },
)
def predict_endpoint(application: CreditApplication):
    """
    Classify a credit application and return the default probability and risk band.
    """
    try:
        result = predict(application)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc