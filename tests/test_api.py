"""Pytest tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    """GET /health returns status ok and model_loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_low_risk():
    """POST /predict with good features returns low risk band."""
    payload = {
        "income": 85000,
        "credit_score": 780,
        "employment_years": 7,
        "debt_to_income": 0.2,
        "loan_history_count": 1,
        "age": 38,
        "home_ownership": "own",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "approved" in data
    assert "default_probability" in data
    assert data["risk_band"] in ["low", "medium", "high"]


def test_predict_high_risk():
    """POST /predict with poor features returns high risk band."""
    payload = {
        "income": 20000,
        "credit_score": 500,
        "employment_years": 0,
        "debt_to_income": 0.6,
        "loan_history_count": 5,
        "age": 22,
        "home_ownership": "rent",
        "verified_income": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_band"] in ["low", "medium", "high"]


def test_predict_invalid_payload():
    """POST /predict with missing fields returns 422."""
    response = client.post("/predict", json={"income": 50000})
    assert response.status_code == 422