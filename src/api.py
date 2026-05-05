"""FastAPI application with credit scoring endpoints."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import load_artifacts
from src.predict import PredictionInput, predict_default

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0"
)

_model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    global _model_loaded
    try:
        load_artifacts()
        _model_loaded = True
    except Exception as e:
        _model_loaded = False
        raise RuntimeError(f"Failed to load model artifacts: {e}")


@app.get(
    "/health",
    response_model=dict,
    summary="Health Check",
    description="Returns service health status and whether the model is loaded."
)
async def health_check() -> JSONResponse:
    """
    Health check endpoint.

    Returns:
        status: "ok" if service is running
        model_loaded: True if model artifacts are successfully loaded
    """
    return JSONResponse(content={
        "status": "ok",
        "model_loaded": _model_loaded
    })


@app.post(
    "/predict",
    response_model=dict,
    summary="Predict Credit Default",
    description="Accepts applicant features and returns default probability with risk band."
)
async def predict(input_data: PredictionInput) -> JSONResponse:
    """
    Credit default prediction endpoint.

    Args:
        input_data: Applicant features for scoring

    Returns:
        approved: Whether the application is approved (prob < 0.35)
        default_probability: Estimated probability of default
        risk_band: Classification into low/medium/high risk
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict_default(input_data)
    return JSONResponse(content=result)