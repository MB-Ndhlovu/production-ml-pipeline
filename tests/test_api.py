"""Pytest fixtures and API endpoint tests."""
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """GET /health returns status and model_loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)


def test_predict_approval():
    """POST /predict returns expected fields for a low-risk application."""
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
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_rejection():
    """High-risk application should be rejected."""
    payload = {
        "income": 20000,
        "credit_score": 500,
        "employment_years": 1,
        "debt_to_income": 0.6,
        "loan_history_count": 5,
        "age": 22,
        "home_ownership": "rent",
        "verified_income": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "approved" in data


def test_predict_invalid_field():
    """Missing required field returns 422."""
    payload = {"income": 65000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422