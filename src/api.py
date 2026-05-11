from fastapi import FastAPI
from src.predict import PredictionRequest, PredictionResponse, predict_default

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
)


@app.get("/health", tags=["health"])
async def health():
    """Check service health and model loading status."""
    from src.model import get_model
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: PredictionRequest):
    """
    Predict credit default probability and risk band.

    Returns the predicted probability of default, risk classification,
    and whether the application is approved based on a 0.5 threshold.
    """
    return predict_default(request)