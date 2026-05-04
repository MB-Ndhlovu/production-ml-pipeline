# Production ML Pipeline with FastAPI

A production-ready credit scoring API using FastAPI with scikit-learn model serving.

## Overview

This project serves a trained credit scoring model via a FastAPI REST API. It loads pre-trained artifacts (model, scaler, feature names) and exposes endpoints for single and batch predictions.

## API Endpoints

### `GET /health`
Health check endpoint. Returns whether the model is loaded and the service is operational.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Predict credit approval and default probability.

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
- `medium`: probability 0.15 – 0.35
- `high`: probability > 0.35

## Local Development

```bash
pip install -r requirements.txt
python run_api.py
```

API available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Running Tests

```bash
pip install pytest httpx
pytest tests/
```

## Deployment

Deploy to any platform that supports Python:
- Railway, Render, Fly.io, Heroku, Docker

Example Docker build:
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_api.py"]
```

## Model Artifacts

The model artifacts (`credit_model.pkl`, `scaler.pkl`, `feature_names.pkl`) are downloaded from the [credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository on startup.