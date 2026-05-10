"""Pytest tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Test that /health returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_endpoint_approval():
    """Test /predict with a low-risk application."""
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


def test_predict_endpoint_validation():
    """Test /predict validates required input."""
    payload = {
        "income": 65000,
        "credit_score": 720,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_risk_bands():
    """Test risk band classification."""
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
    data = response.json()
    prob = data["default_probability"]

    if prob < 0.15:
        assert data["risk_band"] == "low"
    elif prob <= 0.35:
        assert data["risk_band"] == "medium"
    else:
        assert data["risk_band"] == "high"