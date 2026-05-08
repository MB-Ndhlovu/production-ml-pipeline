"""pytest tests for the Prediction API."""
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Synchronous test client."""
    with TestClient(app) as client:
        yield client


def test_health(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid(client):
    """POST /predict with valid data returns 200."""
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


def test_predict_missing_field(client):
    """POST /predict with a missing required field returns 422."""
    payload = {
        "income": 65000,
        # missing credit_score
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_home_ownership(client):
    """POST /predict with an invalid home_ownership value returns 422."""
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


def test_predict_risk_band_low(client):
    """When probability is below 0.15, risk_band should be low."""
    payload = {
        "income": 120000,
        "credit_score": 820,
        "employment_years": 15,
        "debt_to_income": 0.1,
        "loan_history_count": 1,
        "age": 45,
        "home_ownership": "own",
        "verified_income": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_band"] == "low"


def test_predict_risk_band_high(client):
    """When probability is above 0.35, risk_band should be high."""
    payload = {
        "income": 20000,
        "credit_score": 500,
        "employment_years": 0,
        "debt_to_income": 0.6,
        "loan_history_count": 5,
        "age": 22,
        "home_ownership": "rent",
        "verified_income": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_band"] == "high"