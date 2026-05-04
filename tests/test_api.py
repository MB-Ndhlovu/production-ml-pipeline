"""Pytest tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    """Test /health endpoint returns status and model loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "ok"


def test_predict_success():
    """Test /predict with valid input returns approval and probability."""
    payload = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "approved" in data
    assert "default_probability" in data
    assert "risk_band" in data
    assert data["risk_band"] in ("low", "medium", "high")


def test_predict_validation():
    """Test /predict rejects invalid input."""
    payload = {"income": -100}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422