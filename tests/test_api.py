"""Pytest tests for the FastAPI credit scoring endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src import model as model_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Ensure model artifacts are loaded before each test."""
    model_module.load_artifacts()


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_content_type(self):
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestPredictEndpoint:
    def test_predict_low_risk(self):
        payload = {
            "income": 85000,
            "credit_score": 780,
            "employment_years": 8,
            "debt_to_income": 0.15,
            "loan_history_count": 1,
            "age": 40,
            "home_ownership": "own",
            "verified_income": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "approved" in data
        assert "default_probability" in data
        assert "risk_band" in data
        assert data["risk_band"] in ("low", "medium", "high")

    def test_predict_invalid_credit_score(self):
        payload = {
            "income": 50000,
            "credit_score": 200,  # below valid range
            "employment_years": 3,
            "debt_to_income": 0.3,
            "loan_history_count": 1,
            "age": 30,
            "home_ownership": "rent",
            "verified_income": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # validation error

    def test_predict_invalid_income(self):
        payload = {
            "income": -1000,  # negative income
            "credit_score": 700,
            "employment_years": 3,
            "debt_to_income": 0.3,
            "loan_history_count": 1,
            "age": 30,
            "home_ownership": "rent",
            "verified_income": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_field(self):
        payload = {
            "income": 50000,
            # missing credit_score
            "employment_years": 3,
            "debt_to_income": 0.3,
            "loan_history_count": 1,
            "age": 30,
            "home_ownership": "rent",
            "verified_income": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_response_schema(self):
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
        data = response.json()
        assert isinstance(data["approved"], bool)
        assert isinstance(data["default_probability"], float)
        assert 0 <= data["default_probability"] <= 1
        assert data["risk_band"] in ("low", "medium", "high")