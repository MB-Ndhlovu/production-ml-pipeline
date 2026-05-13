"""Model loading utilities for credit risk prediction."""

import joblib
from pathlib import Path

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


def get_model():
    """Return the loaded model, loading if necessary."""
    if _model is None:
        load_artifacts()
    return _model


def get_scaler():
    """Return the loaded scaler, loading if necessary."""
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names():
    """Return the loaded feature names list, loading if necessary."""
    if _feature_names is None:
        load_artifacts()
    return _feature_names


def is_model_loaded() -> bool:
    """Check whether all artifacts are loaded."""
    return _model is not None and _scaler is not None and _feature_names is not None