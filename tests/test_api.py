"""Pytest tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        """Health response should contain 'ok' status."""
        data = response = client.get("/health").json()
        assert data.get("status") == "ok"

    def test_health_contains_model_loaded(self):
        """Health response should contain model_loaded bool."""
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    """Tests for POST /predict."""

    @pytest.fixture
    def valid_payload(self):
        """Standard valid prediction payload."""
        return {
            "income": 65000,
            "credit_score": 720,
            "employment_years": 5,
            "debt_to_income": 0.28,
            "loan_history_count": 2,
            "age": 34,
            "home_ownership": "rent",
            "verified_income": 1,
        }

    def test_predict_returns_200(self, valid_payload):
        """Predict endpoint should return 200 with valid input."""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_returns_required_fields(self, valid_payload):
        """Response should contain approved, default_probability, risk_band."""
        data = client.post("/predict", json=valid_payload).json()
        assert "approved" in data
        assert "default_probability" in data
        assert "risk_band" in data

    def test_approved_is_boolean(self, valid_payload):
        """approved field must be a boolean."""
        data = client.post("/predict", json=valid_payload).json()
        assert isinstance(data["approved"], bool)

    def test_default_probability_is_float(self, valid_payload):
        """default_probability must be a float between 0 and 1."""
        data = client.post("/predict", json=valid_payload).json()
        prob = data["default_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_risk_band_valid_values(self, valid_payload):
        """risk_band must be one of low, medium, high."""
        data = client.post("/predict", json=valid_payload).json()
        assert data["risk_band"] in ("low", "medium", "high")

    def test_risk_band_low_for_low_prob(self):
        """Probability < 0.15 should produce risk_band 'low'."""
        payload = {
            "income": 120000,
            "credit_score": 800,
            "employment_years": 10,
            "debt_to_income": 0.1,
            "loan_history_count": 0,
            "age": 40,
            "home_ownership": "own",
            "verified_income": 1,
        }
        data = client.post("/predict", json=payload).json()
        if data["default_probability"] < 0.15:
            assert data["risk_band"] == "low"

    def test_risk_band_high_for_high_prob(self):
        """Probability > 0.35 should produce risk_band 'high'."""
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
        data = client.post("/predict", json=payload).json()
        if data["default_probability"] > 0.35:
            assert data["risk_band"] == "high"

    def test_predict_rejects_missing_fields(self):
        """Missing required fields should return 422."""
        response = client.post("/predict", json={"income": 50000})
        assert response.status_code == 422

    def test_predict_rejects_invalid_home_ownership(self, valid_payload):
        """Invalid home_ownership value should return 422."""
        valid_payload["home_ownership"] = "invalid_option"
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422