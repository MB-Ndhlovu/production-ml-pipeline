import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

_credit_model = None
_scaler = None
_feature_names = None


def load_artifacts():
    """Load model artifacts from disk."""
    global _credit_model, _scaler, _feature_names
    if _credit_model is None:
        _credit_model = joblib.load(MODEL_DIR / "credit_model.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _credit_model, _scaler, _feature_names


def get_model():
    """Return the loaded credit model."""
    return load_artifacts()[0]


def get_scaler():
    """Return the loaded scaler."""
    return load_artifacts()[1]


def get_feature_names():
    """Return the loaded feature names list."""
    return load_artifacts()[2]