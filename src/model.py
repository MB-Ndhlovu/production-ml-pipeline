"""Model loading and prediction utilities."""

import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load model, scaler, and feature names from disk."""
    global _model, _scaler, _feature_names

    _model = joblib.load(MODEL_PATH / "credit_model.pkl")
    _scaler = joblib.load(MODEL_PATH / "scaler.pkl")
    _feature_names = joblib.load(MODEL_PATH / "feature_names.pkl")

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


def get_feature_names():
    """Get the feature names, loading if necessary."""
    global _feature_names
    if _feature_names is None:
        load_artifacts()
    return _feature_names


def is_model_loaded():
    """Check if all artifacts are loaded."""
    return _model is not None and _scaler is not None and _feature_names is not None