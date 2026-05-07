"""FastAPI application for credit scoring predictions."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .predict import PredictionInput, PredictionOutput, predict_default, get_model

_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    global _model_loaded
    try:
        get_model()
        _model_loaded = True
    except Exception as e:
        _model_loaded = False
        print(f"Warning: model failed to load: {e}")
    yield
    _model_loaded = False


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="Health check",
    response_model=dict,
    responses={200: {"description": "Service is healthy"}},
)
def health():
    """Check if the service and model are up and running."""
    return JSONResponse(
        content={"status": "ok", "model_loaded": _model_loaded}
    )


@app.post(
    "/predict",
    summary="Predict credit default",
    response_model=PredictionOutput,
    responses={
        200: {"description": "Prediction returned successfully"},
        400: {"description": "Invalid input data"},
        500: {"description": "Internal server error"},
    },
)
def predict(input_data: PredictionInput):
    """Predict credit default probability and risk band for a loan application."""
    try:
        result = predict_default(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))