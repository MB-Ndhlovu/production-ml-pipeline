"""Model loading utilities for the credit scoring pipeline."""

import os
from pathlib import Path
from typing import Any, List

import joblib

_MODEL_DIR = Path(__file__).parent.parent / "models"

_credit_model: Any = None
_scaler: Any = None
_feature_names: List[str] = []


def _load_artifacts() -> None:
    """Load all model artifacts from disk."""
    global _credit_model, _scaler, _feature_names
    _credit_model = joblib.load(_MODEL_DIR / "credit_model.pkl")
    _scaler = joblib.load(_MODEL_DIR / "scaler.pkl")
    _feature_names = joblib.load(_MODEL_DIR / "feature_names.pkl")


def get_model():
    """Return the loaded credit_model artifact."""
    if _credit_model is None:
        _load_artifacts()
    return _credit_model


def get_scaler():
    """Return the loaded scaler artifact."""
    if _scaler is None:
        _load_artifacts()
    return _scaler


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names."""
    if not _feature_names:
        _load_artifacts()
    return _feature_names


def is_model_loaded() -> bool:
    """Check whether all artifacts are loaded."""
    try:
        get_model()
        get_scaler()
        get_feature_names()
        return True
    except Exception:
        return False