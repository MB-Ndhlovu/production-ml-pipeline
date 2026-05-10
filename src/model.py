"""Model artifact loading utilities."""

import joblib
import os
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "models"


def load_model():
    """Load the trained credit scoring model."""
    model_path = ARTIFACTS_DIR / "credit_model.pkl"
    return joblib.load(model_path)


def load_scaler():
    """Load the feature scaler."""
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    return joblib.load(scaler_path)


def load_feature_names():
    """Load the expected feature names."""
    names_path = ARTIFACTS_DIR / "feature_names.pkl"
    return joblib.load(names_path)