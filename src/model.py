"""Artifact loading utilities for the credit scoring model."""

import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"


def load_model():
    """Load the trained credit scoring model from models/credit_model.pkl."""
    return joblib.load(MODEL_DIR / "credit_model.pkl")


def load_scaler():
    """Load the feature scaler from models/scaler.pkl."""
    return joblib.load(MODEL_DIR / "scaler.pkl")


def load_feature_names():
    """Load the expected feature names from models/feature_names.pkl."""
    return joblib.load(MODEL_DIR / "feature_names.pkl")