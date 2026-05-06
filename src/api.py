from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .predict import PredictionInput, PredictionOutput, predict_default
from .model import load_artifacts, get_model

app = FastAPI(
    title="Credit Default Prediction API",
    description="Production ML pipeline for credit default predictions with risk classification.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    global _model_loaded
    try:
        load_artifacts()
        get_model()
        _model_loaded = True
    except Exception:
        _model_loaded = False


@app.get(
    "/health",
    summary="Health check",
    description="Check if the API is running and the model is loaded.",
    response_model=dict,
)
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and whether the model artifacts are loaded.
    """
    return JSONResponse(content={"status": "ok", "model_loaded": _model_loaded})


@app.post(
    "/predict",
    summary="Predict credit default",
    description="Submit credit application features to get a default probability prediction and risk assessment.",
    response_model=PredictionOutput,
    responses={
        200: {"description": "Prediction successful"},
        422: {"description": "Validation error"},
    },
)
async def predict(input_data: PredictionInput):
    """
    Predict credit default probability.

    Accepts credit application features and returns the predicted probability
    of default, approval decision, and risk classification.
    """
    return predict_default(input_data)