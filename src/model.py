"""Model loading and inference utilities."""
import joblib
from pathlib import Path
from typing import Optional

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_artifacts() -> bool:
    """Load model, scaler, and feature names from disk."""
    global _model, _scaler, _feature_names
    try:
        _model = joblib.load(MODEL_DIR / "credit_model.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        return True
    except Exception:
        return False


def is_model_loaded() -> bool:
    """Check if model artifacts are loaded."""
    return _model is not None and _scaler is not None and _feature_names is not None


def predict_proba(features: dict) -> float:
    """Predict default probability from raw features."""
    import pandas as pd

    if not is_model_loaded():
        raise RuntimeError("Model artifacts not loaded")

    # Build DataFrame in correct feature order
    df = pd.DataFrame([features])

    # One-hot encode home_ownership
    home_ownership = features.get("home_ownership", "rent")
    for col in _feature_names:
        if col.startswith("home_ownership_"):
            df[col] = 1 if col == f"home_ownership_{home_ownership}" else 0

    # Ensure correct column order
    df = df.reindex(columns=_feature_names, fill_value=0)

    # Scale and predict
    X = _scaler.transform(df)
    prob = _model.predict_proba(X)[0][1]
    return float(prob)