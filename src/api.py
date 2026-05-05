"""FastAPI application for credit default prediction."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .model import load_artifacts, is_model_loaded
from .predict import PredictionInput, PredictionOutput, predict_default


app = FastAPI(
    title="Credit Default Prediction API",
    description="API for predicting credit default probability and risk classification",
    version="1.0.0"
)

_model_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load ML artifacts on startup."""
    global _model_loaded
    try:
        load_artifacts()
        _model_loaded = is_model_loaded()
    except Exception as e:
        _model_loaded = False
        print(f"Failed to load model: {e}")


@app.get("/health", response_model=dict, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the API and whether the model is loaded.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": _model_loaded
        }
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Predict credit default probability.

    Accepts feature values and returns approval decision, default probability,
    and risk band classification.
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predict_default(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))