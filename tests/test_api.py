import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "approved" in data
    assert "default_probability" in data
    assert "risk_band" in data
    assert data["risk_band"] in ["low", "medium", "high"]

def test_predict_response_format():
    """Test that response contains all required fields with correct types."""
    payload = {
        "income": 50000,
        "credit_score": 680,
        "employment_years": 3,
        "debt_to_income": 0.35,
        "loan_history_count": 1,
        "age": 28,
        "home_ownership": "own",
        "verified_income": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data["approved"], bool)
    assert isinstance(data["default_probability"], float)
    assert 0 <= data["default_probability"] <= 1
    assert data["risk_band"] in ["low", "medium", "high"]

def test_predict_invalid_credit_score():
    """Test that invalid credit score is rejected."""
    payload = {
        "income": 65000,
        "credit_score": 999,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_income():
    """Test that negative income is rejected."""
    payload = {
        "income": -1000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_home_ownership_values():
    """Test different home ownership values."""
    for home_type in ["rent", "own", "mortgage", "other"]:
        payload = {
            "income": 65000,
            "credit_score": 720,
            "employment_years": 5,
            "debt_to_income": 0.28,
            "loan_history_count": 2,
            "age": 34,
            "home_ownership": home_type,
            "verified_income": 1
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200