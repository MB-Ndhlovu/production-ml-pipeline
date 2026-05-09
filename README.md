# Production ML Pipeline with FastAPI

A production-ready credit scoring prediction API built with FastAPI.

## Overview

This project exposes a trained credit scoring model via a REST API with health checks, individual predictions, and batch processing support.

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
Make a credit default prediction.

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

**Approval Rule:** `approved = true` if probability < 0.35.

## Installation

```bash
pip install -r requirements.txt
```

## Local Development

```bash
python run_api.py
```

The server starts at `http://localhost:8000`. API docs are available at `http://localhost:8000/docs`.

## Running Tests

```bash
pytest tests/ -v
```

## Batch Prediction

```bash
python src/batch.py --url http://localhost:8000 --input data.csv --output predictions.csv
```

## Deployment

1. Set environment variables for any secrets.
2. Run with uvicorn: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
3. For production, use gunicorn with uvicorn workers.

## Model Artifacts

Model artifacts are downloaded from the [credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository:
- `credit_model.pkl` – trained classifier
- `scaler.pkl` – feature scaler
- `feature_names.pkl` – expected feature names