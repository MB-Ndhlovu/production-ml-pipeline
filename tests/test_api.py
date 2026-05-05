"""Pytest tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import sys.path.insert(0, str(sys.path[0] / ".." ))

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "ok"


def test_predict_endpoint_success():
    """Test the prediction endpoint with valid input."""
    payload = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "approved" in data
    assert "default_probability" in data
    assert "risk_band" in data
    assert data["risk_band"] in ["low", "medium", "high"]
    assert 0 <= data["default_probability"] <= 1


def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid input."""
    payload = {
        "income": -100,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_field():
    """Test the prediction endpoint with missing required field."""
    payload = {
        "income": 65000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422