"""Model loading utilities for credit scoring pipeline."""

import joblib
import os
from typing import Optional

MODEL_URL = "https://raw.githubusercontent.com/MB-Ndhlovu/credit-scoring-pipeline/main/models"

_model = None
_scaler = None
_feature_names = None


def _download_if_missing(filename: str, dest: str) -> None:
    """Download artifact from GitHub if not present."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        import urllib.request
        url = f"{MODEL_URL}/{filename}"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved {filename} to {dest}")


def load_artifacts(models_dir: str = "models") -> None:
    """Load all model artifacts from disk, downloading if necessary."""
    global _model, _scaler, _feature_names

    _download_if_missing("credit_model.pkl", f"{models_dir}/credit_model.pkl")
    _download_if_missing("scaler.pkl", f"{models_dir}/scaler.pkl")
    _download_if_missing("feature_names.pkl", f"{models_dir}/feature_names.pkl")

    _model = joblib.load(f"{models_dir}/credit_model.pkl")
    _scaler = joblib.load(f"{models_dir}/scaler.pkl")
    _feature_names = joblib.load(f"{models_dir}/feature_names.pkl")


def get_model():
    """Return the loaded model artifact."""
    if _model is None:
        load_artifacts()
    return _model


def get_scaler():
    """Return the loaded scaler artifact."""
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_feature_names():
    """Return the loaded feature names artifact."""
    if _feature_names is None:
        load_artifacts()
    return _feature_names


def is_model_loaded() -> bool:
    """Check whether all artifacts are loaded."""
    return _model is not None and _scaler is not None and _feature_names is not None