"""FastAPI application for credit scoring API."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import load_artifacts, is_model_loaded
from src.predict import PredictionInput, PredictionOutput, make_prediction

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
)

_loaded = False


@app.on_event("startup")
async def startup():
    """Load model artifacts on startup."""
    global _loaded
    _loaded = load_artifacts()


@app.get("/health", tags=["Health"])
async def health():
    """
    Health check endpoint.

    Returns whether the service is operational and the model is loaded.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": _loaded,
        }
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Predict credit approval and default probability.

    Accepts applicant features and returns approval decision,
    probability of default, and risk classification.
    """
    if not _loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = make_prediction(input_data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))