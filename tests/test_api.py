"""Tests for the Credit Scoring API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.model import credit_model


@pytest.fixture(autouse=True)
def load_model():
    """Ensure model is loaded before each test."""
    if not credit_model.is_loaded:
        credit_model.load()
    yield


client = TestClient(app)


def test_health():
    """GET /health returns correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_success():
    """POST /predict returns correct schema for valid input."""
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
    assert 0.0 <= data["default_probability"] <= 1.0


def test_predict_schema():
    """Response contains all required fields."""
    payload = {
        "income": 50000,
        "credit_score": 600,
        "employment_years": 2,
        "debt_to_income": 0.35,
        "loan_history_count": 1,
        "age": 25,
        "home_ownership": "rent",
        "verified_income": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["approved"], bool)
    assert isinstance(data["default_probability"], float)
    assert isinstance(data["risk_band"], str)


def test_predict_invalid_field():
    """POST /predict returns 422 for missing required fields."""
    payload = {"income": 65000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
