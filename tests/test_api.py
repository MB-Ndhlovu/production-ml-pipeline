"""Test suite for the Credit Scoring API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestHealth:
    """Tests for the GET /health endpoint."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestPredict:
    """Tests for the POST /predict endpoint."""

    def test_predict_low_risk(self):
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

    def test_predict_returns_valid_probability(self):
        payload = {
            "income": 50000,
            "credit_score": 600,
            "employment_years": 2,
            "debt_to_income": 0.4,
            "loan_history_count": 5,
            "age": 25,
            "home_ownership": "rent",
            "verified_income": 0,
        }
        response = client.post("/predict", json=payload)
        data = response.json()
        assert 0 <= data["default_probability"] <= 1

    def test_predict_invalid_credit_score(self):
        payload = {
            "income": 50000,
            "credit_score": 900,  # out of range (> 850)
            "employment_years": 2,
            "debt_to_income": 0.4,
            "loan_history_count": 5,
            "age": 25,
            "home_ownership": "rent",
            "verified_income": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # FastAPI validation error

    def test_predict_missing_field(self):
        payload = {
            "income": 50000,
            # missing credit_score
            "employment_years": 2,
            "debt_to_income": 0.4,
            "loan_history_count": 5,
            "age": 25,
            "home_ownership": "rent",
            "verified_income": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422