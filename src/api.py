from fastapi import FastAPI, HTTPException
from .predict import PredictInput, PredictOutput, predict
from .model import load_artifacts, get_model

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0"
)

_model_loaded = False


@app.on_event("startup")
def startup_event():
    """Load model artifacts on startup."""
    global _model_loaded
    try:
        load_artifacts()
        _model_loaded = True
    except Exception as e:
        _model_loaded = False
        raise RuntimeError(f"Failed to load model artifacts: {e}")


@app.get("/health", response_model=dict, tags=["Health"])
def health():
    """
    Health check endpoint.

    Returns the current status of the service and whether the model is loaded.
    """
    return {
        "status": "ok",
        "model_loaded": _model_loaded
    }


@app.post("/predict", response_model=PredictOutput, tags=["Predictions"])
def predict_endpoint(input_data: PredictInput):
    """
    Predict credit default probability.

    Accepts applicant features and returns an approval decision with
    probability and risk band.
    """
    try:
        return predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))