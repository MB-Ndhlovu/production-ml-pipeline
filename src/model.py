"""Model artifact loading from disk."""
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_model():
    """Load the pickled logistic regression model."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "credit_model.pkl")
    return _model


def load_scaler():
    """Load the fitted StandardScaler."""
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return _scaler


def load_feature_names():
    """Load the expected feature column names."""
    global _feature_names
    if _feature_names is None:
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _feature_names


def is_model_loaded():
    """Check whether all artifacts are loaded."""
    try:
        load_model()
        load_scaler()
        load_feature_names()
        return True
    except Exception:
        return False