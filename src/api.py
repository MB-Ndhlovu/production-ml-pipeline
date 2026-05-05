import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .model import get_model_artifacts
from .predict import PredictionInput, PredictionOutput, get_risk_band, is_approved, prepare_features

_model = None
_scaler = None
_feature_names = None

def _load_artifacts():
    """Load model artifacts once at module import."""
    global _model, _scaler, _feature_names
    if _model is None:
        try:
            _model, _scaler, _feature_names = get_model_artifacts()
        except FileNotFoundError:
            _model = None

_load_artifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    yield

def get_model():
    """Get loaded model artifacts."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model, _scaler, _feature_names

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and whether the model is loaded.
    """
    model_loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict credit default probability.
    
    Accepts applicant features and returns:
    - approved: Whether the application should be approved
    - default_probability: Predicted probability of default (0-1)
    - risk_band: Classification as 'low', 'medium', or 'high'
    """
    model, scaler, feature_names = get_model()
    
    features = prepare_features(input_data)
    
    feature_array = [features[name] for name in feature_names]
    
    scaled_features = scaler.transform([feature_array])
    
    probability = model.predict_proba(scaled_features)[0][1]
    
    risk_band = get_risk_band(probability)
    approved = is_approved(probability)
    
    return PredictionOutput(
        approved=approved,
        default_probability=round(probability, 4),
        risk_band=risk_band
    )