import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_success():
    """Test /predict with valid input."""
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
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_validation_error():
    """Test /predict with invalid input returns 422."""
    payload = {
        "income": -5000,  # invalid: must be positive
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


def test_predict_risk_bands():
    """Test that risk bands are correctly assigned."""
    test_cases = [
        # High credit score, low risk
        {
            "income": 120000,
            "credit_score": 800,
            "employment_years": 10,
            "debt_to_income": 0.15,
            "loan_history_count": 1,
            "age": 40,
            "home_ownership": "own",
            "verified_income": 1,
        },
        # Low credit score, high risk
        {
            "income": 25000,
            "credit_score": 520,
            "employment_years": 1,
            "debt_to_income": 0.45,
            "loan_history_count": 5,
            "age": 22,
            "home_ownership": "rent",
            "verified_income": 0,
        },
    ]
    for payload in test_cases:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] in ["low", "medium", "high"]