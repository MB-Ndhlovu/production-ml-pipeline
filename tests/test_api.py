import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_model():
    """Ensure model is loaded for tests."""
    from src.model import load_artifacts
    load_artifacts()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self):
        """Test that health endpoint returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_response_schema(self):
        """Test health response has correct schema."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_valid_response(self):
        """Test prediction returns all required fields."""
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

    def test_predict_approved_field_is_bool(self):
        """Test that approved field is boolean."""
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
        data = response.json()
        assert isinstance(data["approved"], bool)

    def test_predict_probability_is_float(self):
        """Test that probability is a float between 0 and 1."""
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
        data = response.json()
        prob = data["default_probability"]
        assert isinstance(prob, float)
        assert 0 <= prob <= 1

    def test_predict_risk_band_is_valid(self):
        """Test that risk_band is one of low, medium, high."""
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
        data = response.json()
        assert data["risk_band"] in ["low", "medium", "high"]

    def test_predict_high_risk_input(self):
        """Test prediction with high-risk input."""
        payload = {
            "income": 20000,
            "credit_score": 500,
            "employment_years": 0,
            "debt_to_income": 0.5,
            "loan_history_count": 5,
            "age": 22,
            "home_ownership": "rent",
            "verified_income": 0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] in ["low", "medium", "high"]

    def test_predict_invalid_input(self):
        """Test prediction with invalid input returns 422."""
        payload = {
            "income": -1000,
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