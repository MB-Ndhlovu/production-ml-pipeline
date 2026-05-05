import pytest
from httpx import AsyncClient
from src.api import app


@pytest.mark.asyncio
async def test_health():
    """Test the health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_predict():
    """Test the prediction endpoint."""
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
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "approved" in data
    assert "default_probability" in data
    assert "risk_band" in data
    assert data["risk_band"] in ["low", "medium", "high"]