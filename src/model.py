import joblib
from pathlib import Path
from typing import Optional, List

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load all model artifacts from disk."""
    global _model, _scaler, _feature_names
    
    _model = joblib.load(MODEL_DIR / "credit_model.pkl")
    _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    
    return _model, _scaler, _feature_names


def get_model():
    """Get the loaded model, loading if necessary."""
    global _model
    if _model is None:
        load_artifacts()
    return _model


def get_scaler():
    """Get the loaded scaler, loading if necessary."""
    global _scaler
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names() -> List[str]:
    """Get the feature names, loading if necessary."""
    global _feature_names
    if _feature_names is None:
        load_artifacts()
    return _feature_names


def is_model_loaded() -> bool:
    """Check if model artifacts are loaded."""
    return _model is not None and _scaler is not None and _feature_names is not None