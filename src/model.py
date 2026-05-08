"""Model loading and prediction utilities."""

import os
from pathlib import Path

import joblib
import requests

MODEL_BASE_URL = "https://raw.githubusercontent.com/MB-Ndhlovu/credit-scoring-pipeline/main/models"
MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_scaler = None
_feature_names = None


def _download_artifacts():
    """Download model artifacts from the credit-scoring-pipeline repo."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = ["credit_model.pkl", "scaler.pkl", "feature_names.pkl"]
    for name in artifacts:
        dest = MODEL_DIR / name
        if not dest.exists():
            resp = requests.get(f"{MODEL_BASE_URL}/{name}", timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)


def load_model_artifacts():
    """Load model, scaler, and feature names from disk."""
    global _model, _scaler, _feature_names

    if _model is None:
        _download_artifacts()
        _model = joblib.load(MODEL_DIR / "credit_model.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return _model, _scaler, _feature_names


def predict(features: dict) -> tuple[float, bool, str]:
    """
    Run prediction on a feature dict.

    Returns:
        tuple: (default_probability, approved, risk_band)
    """
    model, scaler, feature_names = load_model_artifacts()

    import pandas as pd

    df = pd.DataFrame([features])

    # One-hot encode home_ownership if present
    if "home_ownership" in df.columns:
        home_ownership = df["home_ownership"].iloc[0]
        for col in feature_names:
            if col.startswith("home_ownership_"):
                df[col] = 1 if col == f"home_ownership_{home_ownership}" else 0
        df = df.drop(columns=["home_ownership"])

    # Ensure correct column order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale and predict
    X = scaler.transform(df)
    prob = float(model.predict_proba(X)[0, 1])

    approved = prob < 0.35
    if prob < 0.15:
        risk_band = "low"
    elif prob <= 0.35:
        risk_band = "medium"
    else:
        risk_band = "high"

    return prob, approved, risk_band