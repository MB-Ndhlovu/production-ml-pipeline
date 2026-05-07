import pytest
from httpx import AsyncClient, ASGITransport
from src.api import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_predict():
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "approved" in data
    assert "default_probability" in data
    assert "risk_band" in data
    assert data["risk_band"] in ("low", "medium", "high")
    assert 0 <= data["default_probability"] <= 1