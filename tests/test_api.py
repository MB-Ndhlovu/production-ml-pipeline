"""Pytest test suite for the credit scoring API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_valid():
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
    assert data["risk_band"] in ("low", "medium", "high")


def test_predict_invalid_credit_score():
    payload = {
        "income": 65000,
        "credit_score": 900,  # out of range
        "employment_years": 5,
        "debt_to_income": 0.28,
        "loan_history_count": 2,
        "age": 34,
        "home_ownership": "rent",
        "verified_income": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    payload = {
        "income": 65000,
        "credit_score": 720,
        # missing fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422