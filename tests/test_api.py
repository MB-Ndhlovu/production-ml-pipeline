"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestPredict:
    def test_predict_returns_required_fields(self):
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

    def test_predict_validates_home_ownership(self):
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

    def test_predict_requires_all_fields(self):
        response = client.post("/predict", json={"income": 50000})
        assert response.status_code == 422