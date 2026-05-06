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


def test_predict_success():
    """Test successful prediction request."""
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
    assert data["risk_band"] in ["low", "medium", "high"]
    assert 0 <= data["default_probability"] <= 1


def test_predict_validation():
    """Test prediction with invalid data."""
    payload = {
        "income": "not_a_number",
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


def test_predict_all_home_ownership():
    """Test prediction with different home ownership values."""
    for home_ownership in ["rent", "own", "mortgage", "other"]:
        payload = {
            "income": 65000,
            "credit_score": 720,
            "employment_years": 5,
            "debt_to_income": 0.28,
            "loan_history_count": 2,
            "age": 34,
            "home_ownership": home_ownership,
            "verified_income": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] in ["low", "medium", "high"]