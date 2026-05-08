"""Model loading utilities."""
import joblib
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).parent.parent / "models"

_model: Any = None
_scaler: Any = None
_feature_names: list[str] | None = None


def load_artifacts() -> tuple[Any, Any, list[str]]:
    """Load and return (model, scaler, feature_names) from models/."""
    global _model, _scaler, _feature_names
    _model = joblib.load(MODEL_DIR / "credit_model.pkl")
    _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _model, _scaler, _feature_names


def get_model() -> Any:
    """Return the loaded model, loading if necessary."""
    global _model
    if _model is None:
        load_artifacts()
    return _model


def get_scaler() -> Any:
    """Return the loaded scaler, loading if necessary."""
    global _scaler
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names() -> list[str]:
    """Return the loaded feature names, loading if necessary."""
    global _feature_names
    if _feature_names is None:
        load_artifacts()
    return _feature_names