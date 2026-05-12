"""FastAPI application for credit scoring predictions."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import load_artifacts, is_model_loaded
from src.predict import CreditScoreInput, PredictionOutput, predict

# Load model artifacts on startup
load_artifacts()

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
)


@app.get("/health", response_model=dict, tags=["health"])
async def health():
    """
    Health check endpoint.

    Returns the service status and whether the model artifacts are loaded.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": is_model_loaded(),
        }
    )


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
async def make_prediction(input_data: CreditScoreInput):
    """
    Predict credit default risk.

    Accepts applicant features and returns the approval decision,
    default probability, and risk band.
    """
    try:
        result = predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))