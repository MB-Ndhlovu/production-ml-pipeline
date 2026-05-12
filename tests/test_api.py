import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_approval():
    """Test a low-risk prediction that should be approved."""
    payload = {
        "income": 85000,
        "credit_score": 780,
        "employment_years": 7,
        "debt_to_income": 0.15,
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
    assert "risk_band" in data
    assert data["risk_band"] in ["low", "medium", "high"]


def test_predict_rejection():
    """Test a high-risk prediction that should be rejected."""
    payload = {
        "income": 25000,
        "credit_score": 520,
        "employment_years": 0.5,
        "debt_to_income": 0.6,
        "loan_history_count": 5,
        "age": 22,
        "home_ownership": "rent",
        "verified_income": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["default_probability"] > 0.35
    assert data["risk_band"] == "high"


def test_predict_invalid_data():
    """Test prediction with invalid data returns 422."""
    payload = {
        "income": -5000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
