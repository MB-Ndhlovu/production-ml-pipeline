"""Model loading and artifact management."""

import joblib
from pathlib import Path
from typing import Optional

MODEL_DIR = Path(__file__).parent.parent / "models"


def load_artifacts():
    """Load model, scaler, and feature names from disk."""
    model = joblib.load(MODEL_DIR / "credit_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, scaler, feature_names


class CreditModel:
    """Wrapper around the credit scoring model artifacts."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._loaded = False

    def load(self) -> bool:
        """Load all artifacts. Returns True on success."""
        try:
            self.model, self.scaler, self.feature_names = load_artifacts()
            self._loaded = True
            return True
        except Exception as e:
            print(f"Failed to load model artifacts: {e}")
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Global singleton
credit_model = CreditModel()
