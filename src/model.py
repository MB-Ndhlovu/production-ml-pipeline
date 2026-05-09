import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_credit_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load model artifacts from the models directory."""
    global _credit_model, _scaler, _feature_names
    _credit_model = joblib.load(os.path.join(MODEL_DIR, "credit_model.pkl"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return _credit_model, _scaler, _feature_names


def get_model():
    if _credit_model is None:
        load_artifacts()
    return _credit_model


def get_scaler():
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names():
    if _feature_names is None:
        load_artifacts()
    return _feature_names


def is_model_loaded():
    try:
        get_model()
        return True
    except Exception:
        return False