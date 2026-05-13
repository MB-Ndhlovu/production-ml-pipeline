import os
from pathlib import Path
import joblib
import numpy as np

MODEL_DIR = Path(__file__).parent.parent / "models"

_credit_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load all ML artifacts from the models directory."""
    global _credit_model, _scaler, _feature_names

    model_path = MODEL_DIR / "credit_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    feature_names_path = MODEL_DIR / "feature_names.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found: {feature_names_path}")

    _credit_model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    _feature_names = joblib.load(feature_names_path)

    return _credit_model, _scaler, _feature_names


def get_model():
    """Get the credit model, loading if necessary."""
    global _credit_model
    if _credit_model is None:
        load_artifacts()
    return _credit_model


def get_scaler():
    """Get the scaler, loading if necessary."""
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
    """Check if model is loaded."""
    global _credit_model
    return _credit_model is not None