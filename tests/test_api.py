import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    """Test GET /health returns status and model_loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_predict_success():
    """Test POST /predict with valid input returns expected fields."""
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
    assert isinstance(data["approved"], bool)
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert "risk_band" in data
    assert data["risk_band"] in ("low", "medium", "high")


def test_predict_risk_band_low():
    """Test low-risk applicant returns low band."""
    payload = {
        "income": 120000,
        "credit_score": 800,
        "employment_years": 10,
        "debt_to_income": 0.1,
        "loan_history_count": 1,
        "age": 40,
        "home_ownership": "own",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_band"] == "low"


def test_predict_missing_field():
    """Test POST /predict with missing required field returns 422."""
    payload = {
        "income": 65000,
        "credit_score": 720,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_home_ownership():
    """Test POST /predict with invalid home_ownership returns 422."""
    payload = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "invalid",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422