# Production ML Pipeline with FastAPI

A production-ready credit scoring API built with FastAPI, serving predictions from a trained scikit-learn model.

## Overview

This service loads a pre-trained credit scoring model and exposes it via a REST API for real-time predictions. It supports:
- Single prediction via POST `/predict`
- Batch prediction via script
- Health checks for deployment monitoring

## API Endpoints

### GET /health
Health check endpoint. Returns service status and model load state.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict
Submit a credit application for scoring.

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

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Running Tests

```bash
pytest tests/ -v
```

## Deployment

The API is stateless and can be deployed to any platform that supports Python:
- Railway, Render, Fly.io (recommended for simplicity)
- Docker container
- Kubernetes via uvicorn/gunicorn

## Model Artifacts

The model expects three artifacts in the `models/` directory:
- `credit_model.pkl` — trained classifier
- `scaler.pkl` — fitted StandardScaler
- `feature_names.pkl` — list of feature names in order

These are downloaded from the credit-scoring-pipeline repository on first setup.