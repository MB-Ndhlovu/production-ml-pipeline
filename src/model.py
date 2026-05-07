"""Model loading utilities."""
import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model():
    """Load the trained credit scoring model."""
    path = os.path.join(MODEL_DIR, "credit_model.pkl")
    return joblib.load(path)


def load_scaler():
    """Load the feature scaler."""
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(path)


def load_feature_names():
    """Load the expected feature names."""
    path = os.path.join(MODEL_DIR, "feature_names.pkl")
    return joblib.load(path)