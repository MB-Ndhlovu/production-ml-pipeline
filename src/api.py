"""FastAPI application for the credit scoring prediction API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import load_model_artifacts
from src.predict import PredictInput, PredictOutput, make_prediction

_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global _model_loaded
    try:
        load_model_artifacts()
        _model_loaded = True
    except Exception as e:
        _model_loaded = False
        raise RuntimeError(f"Failed to load model artifacts: {e}") from e
    yield
    _model_loaded = False


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit approval predictions.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="Health check",
    response_model=dict,
    responses={200: {"description": "Service is healthy"}},
)
def health_check() -> dict:
    """
    Health check endpoint.

    Returns the current status of the service and whether the model is loaded.
    """
    return JSONResponse(
        content={"status": "ok", "model_loaded": _model_loaded}
    )


@app.post(
    "/predict",
    summary="Make a credit approval prediction",
    response_model=PredictOutput,
    responses={
        200: {"description": "Prediction result"},
        400: {"description": "Invalid input"},
        500: {"description": "Model or server error"},
    },
)
def predict(input_data: PredictInput) -> PredictOutput:
    """
    Make a credit approval prediction.

    Accepts applicant features and returns an approval decision,
    default probability, and risk band.

    **Risk bands:**
    - `low`: probability < 0.15
    - `medium`: 0.15 <= probability <= 0.35
    - `high`: probability > 0.35
    """
    try:
        result = make_prediction(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e