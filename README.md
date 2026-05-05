# Production ML Pipeline with FastAPI

Credit scoring prediction API using a trained scikit-learn model.

## Overview

This project provides a FastAPI-based REST API for credit default prediction. It loads a pre-trained model and scaler from the `credit-scoring-pipeline` repository and exposes endpoints for single and batch predictions.

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
- `medium`: probability 0.15 - 0.35
- `high`: probability > 0.35

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python run_api.py
```

API available at: http://localhost:8000

## Run Tests

```bash
pytest tests/ -v
```

## Deployment

The API can be deployed to any platform that supports Python:
- Railway
- Render
- Fly.io
- Docker