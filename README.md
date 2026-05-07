# Production ML Pipeline with FastAPI

A production-ready credit scoring API built with FastAPI, scikit-learn, and Pydantic.

## Overview

This project provides a REST API for credit default prediction. It loads a pre-trained model and scaler from artifacts and exposes endpoints for single and batch predictions.

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Predict credit default probability.

**Request Body:**
```json
{
  "income": 65000,
  "credit_score": 720,
  "employment_years": 5,
  "debt_to_income": 0.28,
  "loan_history_count": 2,
  "age": 34,
  "home_ownership": "rent",
  "verified_income": 1
}
```

**Response:**
```json
{
  "approved": true,
  "default_probability": 0.12,
  "risk_band": "low"
}
```

**Risk Bands:**
- `low`: probability < 0.15
- `medium`: 0.15 <= probability <= 0.35
- `high`: probability > 0.35

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python run_api.py

# Run tests
pytest tests/
```

## Deployment

The API can be deployed to any platform that supports Python ASGI apps (UVicorn).

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── run_api.py
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   ├── api.py
│   └── batch.py
└── tests/
    ├── __init__.py
    └── test_api.py
```