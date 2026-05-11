# Production ML Pipeline

A FastAPI-based credit scoring API for real-time inference and batch predictions.

## Overview

This service loads a trained credit scoring model and provides:
- Real-time single prediction via REST API
- Batch prediction via script
- Health check endpoint for monitoring

## Installation

```bash
pip install -r requirements.txt
```

## Run the API

```bash
python run_api.py
```

The server starts on `http://localhost:8000`.

## API Endpoints

### GET /health

Health check. Returns model loading status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Make a credit approval prediction.

**Request body:**
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

**Risk bands:**
- `low`: probability < 0.15
- `medium`: probability 0.15–0.35
- `high`: probability > 0.35

## Run Tests

```bash
pytest tests/ -v
```

## Deployment

The API can be containerized or deployed to any Python-capable host:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

For production, consider:
- Running behind a reverse proxy (nginx)
- Enabling CORS middleware if needed
- Adding authentication
- Using gunicorn for multi-worker deployment

## Batch Predictions

Use `src/batch.py` to score multiple applicants via the API:

```bash
python src/batch.py --url http://localhost:8000 --input data.csv --output results.csv
```
