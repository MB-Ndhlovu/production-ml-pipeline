import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load model artifacts from the models directory."""
    global _model, _scaler, _feature_names

    if _model is None:
        _model = joblib.load(MODEL_DIR / "credit_model.pkl")
    if _scaler is None:
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    if _feature_names is None:
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return _model, _scaler, _feature_names


def get_model():
    """Get the loaded model, loading if necessary."""
    if _model is None:
        load_artifacts()
    return _model


def get_scaler():
    """Get the loaded scaler, loading if necessary."""
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names():
    """Get the feature names, loading if necessary."""
    if _feature_names is None:
        load_artifacts()
    return _feature_names