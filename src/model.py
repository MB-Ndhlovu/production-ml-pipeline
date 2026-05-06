"""Model loading utilities for credit scoring pipeline."""

import joblib
from pathlib import Path
from typing import Optional

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load all model artifacts from disk."""
    global _model, _scaler, _feature_names

    if _model is None:
        _model = joblib.load(MODEL_DIR / "credit_model.pkl")
    if _scaler is None:
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    if _feature_names is None:
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return _model, _scaler, _feature_names


def get_model():
    """Get the loaded model artifact."""
    model, _, _ = load_artifacts()
    return model


def get_scaler():
    """Get the loaded scaler artifact."""
    _, scaler, _ = load_artifacts()
    return scaler


def get_feature_names():
    """Get the loaded feature names."""
    _, _, feature_names = load_artifacts()
    return feature_names


def is_model_loaded() -> bool:
    """Check if all model artifacts are loaded."""
    try:
        load_artifacts()
        return True
    except Exception:
        return False