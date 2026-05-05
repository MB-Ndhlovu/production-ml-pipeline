import os
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_model():
    """Load the trained credit scoring model."""
    model_path = os.path.join(MODELS_DIR, "credit_model.pkl")
    return joblib.load(model_path)

def load_scaler():
    """Load the feature scaler."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    return joblib.load(scaler_path)

def load_feature_names():
    """Load the expected feature names."""
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    return joblib.load(feature_names_path)

def get_model_artifacts():
    """Load all model artifacts."""
    model = load_model()
    scaler = load_scaler()
    feature_names = load_feature_names()
    return model, scaler, feature_names