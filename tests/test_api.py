"""Pytest tests for the credit scoring API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.model import is_model_loaded


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is is_model_loaded()


def test_predict_success(client):
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
    assert 0 <= data["default_probability"] <= 1


def test_predict_validation_error(client):
    payload = {
        "income": 65000,
        "credit_score": 720,
        # missing required fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_credit_score(client):
    payload = {
        "income": 65000,
        "credit_score": 9999,  # out of range
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422