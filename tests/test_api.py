import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["is_model_loaded"], bool)


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
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "approved" in data
    assert "default_probability" in data
    assert data["risk_band"] in ("low", "medium", "high")
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_high_prob():
    payload = {
        "income": 20000,
        "credit_score": 500,
        "employment_years": 0,
        "debt_to_income": 0.8,
        "loan_history_count": 5,
        "age": 22,
        "home_ownership": "rent",
        "verified_income": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_band"] == "high"
    assert data["approved"] is False


def test_predict_invalid():
    payload = {"income": -100}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422