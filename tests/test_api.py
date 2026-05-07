"""Pytest tests for the credit scoring API."""
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_approval():
    """Test /predict with a low-risk applicant."""
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


def test_predict_rejection():
    """Test /predict with a high-risk applicant."""
    payload = {
        "income": 20000,
        "credit_score": 450,
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
    assert data["approved"] is False


def test_predict_invalid():
    """Test /predict with missing required fields."""
    payload = {"income": 65000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422