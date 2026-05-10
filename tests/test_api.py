"""Pytest suite for the Credit Scoring API."""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture(scope="module", autouse=True)
def init_model():
    """Initialize model artifacts before tests run."""
    import src.api as api_module
    from src.model import load_model, load_scaler, load_feature_names

    api_module.model = load_model()
    api_module.scaler = load_scaler()
    api_module.feature_names = load_feature_names()
    api_module.model_loaded = True


client = TestClient(app)


def test_health_check():
    """GET /health returns ok status and model loaded flag."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_correct_schema():
    """POST /predict returns all required fields with correct types."""
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
    assert "risk_band" in data
    assert isinstance(data["approved"], bool)
    assert isinstance(data["default_probability"], float)
    assert data["risk_band"] in ("low", "medium", "high")


def test_predict_invalid_score():
    """Out-of-range credit score returns a validation error."""
    payload = {
        "income": 50000,
        "credit_score": 200,
        "employment_years": 3,
        "debt_to_income": 0.3,
        "loan_history_count": 1,
        "age": 30,
        "home_ownership": "rent",
        "verified_income": 1,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_home_ownership():
    """Invalid home_ownership value returns a validation error."""
    payload = {
        "income": 50000,
        "credit_score": 650,
        "employment_years": 3,
        "debt_to_income": 0.3,
        "loan_history_count": 1,
        "age": 30,
        "home_ownership": "invalid",
        "verified_income": 1,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_risk_bands_are_deterministic():
    """Same input always produces the same risk band."""
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
    resp1 = client.post("/predict", json=payload)
    resp2 = client.post("/predict", json=payload)
    assert resp1.json()["risk_band"] == resp2.json()["risk_band"]