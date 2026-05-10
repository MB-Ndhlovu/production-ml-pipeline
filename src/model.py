"""Model loading utilities for credit scoring pipeline."""

import os
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_model = None
_scaler = None
_feature_names = None


def get_model():
    """Return the loaded sklearn model, loading on first call."""
    global _model
    if _model is None:
        _model = joblib.load(MODELS_DIR / "credit_model.pkl")
    return _model


def get_scaler():
    """Return the loaded StandardScaler, loading on first call."""
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return _scaler


def get_feature_names():
    """Return the list of feature names used during training."""
    global _feature_names
    if _feature_names is None:
        _feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return _feature_names


def is_model_loaded() -> bool:
    """Check whether all model artifacts are loaded successfully."""
    try:
        get_model()
        get_scaler()
        get_feature_names()
        return True
    except Exception:
        return False