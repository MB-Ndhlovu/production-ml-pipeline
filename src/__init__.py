"""Production ML Pipeline source package."""

from src.model import load_model_artifacts, predict

__all__ = ["load_model_artifacts", "predict"]