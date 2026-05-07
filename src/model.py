import joblib
import os
from pathlib import Path

MODEL_URL = "https://raw.githubusercontent.com/MB-Ndhlovu/credit-scoring-pipeline/main/models"

_models_dir = Path(__file__).parent.parent / "models"
_models_dir.mkdir(exist_ok=True)


def _download(url: str, dest: Path) -> None:
    if not dest.exists():
        import urllib.request
        urllib.request.urlretrieve(url, dest)


def load_artifacts():
    _download(f"{MODEL_URL}/credit_model.pkl", _models_dir / "credit_model.pkl")
    _download(f"{MODEL_URL}/scaler.pkl", _models_dir / "scaler.pkl")
    _download(f"{MODEL_URL}/feature_names.pkl", _models_dir / "feature_names.pkl")

    model = joblib.load(_models_dir / "credit_model.pkl")
    scaler = joblib.load(_models_dir / "scaler.pkl")
    feature_names = joblib.load(_models_dir / "feature_names.pkl")
    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()