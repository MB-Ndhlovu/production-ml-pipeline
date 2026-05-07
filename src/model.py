import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def _patch_model(model):
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    if not hasattr(model, "n_features_in_"):
        model.n_features_in_ = 18
    return model


def load_artifacts():
    """Load model, scaler, and feature names from disk."""
    global _model, _scaler, _feature_names
    _model = _patch_model(joblib.load(MODEL_DIR / "credit_model.pkl"))
    _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _model, _scaler, _feature_names


def get_model():
    if _model is None:
        load_artifacts()
    return _model


def get_scaler():
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names():
    if _feature_names is None:
        load_artifacts()
    return _feature_names