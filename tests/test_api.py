import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict():
    payload = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1,
        "loan_amount": 10000,
        "interest_rate": 0.10,
        "num_credit_lines": 2,
        "delinquency_2yrs": 0,
        "loan_purpose": "other"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "approved" in data
    assert "default_probability" in data
    assert "risk_band" in data
    assert data["risk_band"] in ["low", "medium", "high"]
    assert 0 <= data["default_probability"] <= 1


def test_predict_home_ownership_values():
    for ho in ["rent", "own", "mortgage"]:
        payload = {
            "income": 50000,
            "credit_score": 700,
            "employment_years": 3,
            "debt_to_income": 0.25,
            "loan_history_count": 1,
            "age": 30,
            "home_ownership": ho,
            "verified_income": 1,
            "loan_amount": 5000,
            "interest_rate": 0.08,
            "num_credit_lines": 1,
            "delinquency_2yrs": 0,
            "loan_purpose": "other"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200