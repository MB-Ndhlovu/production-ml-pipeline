import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def load_model():
    """Load model artifacts from disk."""
    global _model, _scaler, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_DIR / "credit_model.pkl")
    return _model


def load_scaler():
    """Load scaler artifact from disk."""
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return _scaler


def load_feature_names():
    """Load feature names from disk."""
    global _feature_names
    if _feature_names is None:
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _feature_names


def download_artifacts():
    """Download model artifacts from the credit-scoring-pipeline repo."""
    import urllib.request
    artifacts = ["credit_model.pkl", "scaler.pkl", "feature_names.pkl"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/MB-Ndhlovu/credit-scoring-pipeline/main/models/"
    for artifact in artifacts:
        dest = MODEL_DIR / artifact
        if not dest.exists():
            urllib.request.urlretrieve(base_url + artifact, dest)
            print(f"Downloaded {artifact}")
        else:
            print(f"{artifact} already exists, skipping")


if __name__ == "__main__":
    download_artifacts()
    load_model()
    load_scaler()
    load_feature_names()
    print("All artifacts loaded successfully")