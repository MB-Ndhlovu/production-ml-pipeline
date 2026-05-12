"""Tests for the FastAPI /predict and /health endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

SAMPLE_APPLICATION = {
    "income": 65000,
    "credit_score": 720,
    "employment_years": 5,
    "debt_to_income": 0.28,
    "loan_history_count": 2,
    "age": 34,
    "home_ownership": "rent",
    "verified_income": 1,
}


class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_contains_model_loaded(self):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestPredict:
    def test_predict_returns_required_fields(self):
        response = client.post("/predict", json=SAMPLE_APPLICATION)
        assert response.status_code == 200
        data = response.json()
        assert "approved" in data
        assert "default_probability" in data
        assert "risk_band" in data

    def test_predict_approved_is_bool(self):
        response = client.post("/predict", json=SAMPLE_APPLICATION)
        data = response.json()
        assert isinstance(data["approved"], bool)

    def test_predict_probability_is_float(self):
        response = client.post("/predict", json=SAMPLE_APPLICATION)
        data = response.json()
        assert isinstance(data["default_probability"], float)
        assert 0.0 <= data["default_probability"] <= 1.0

    def test_predict_risk_band_valid(self):
        response = client.post("/predict", json=SAMPLE_APPLICATION)
        data = response.json()
        assert data["risk_band"] in ("low", "medium", "high")

    def test_predict_invalid_credit_score_rejected(self):
        bad = dict(SAMPLE_APPLICATION, credit_score=999)
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_negative_income_rejected(self):
        bad = dict(SAMPLE_APPLICATION, income=-1000)
        response = client.post("/predict", json=bad)
        assert response.status_code == 422