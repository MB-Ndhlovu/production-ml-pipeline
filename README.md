# Production ML Pipeline with FastAPI

A production-ready credit scoring prediction API built with FastAPI.

## Overview

This project provides a REST API for credit default prediction, featuring:
- Real-time credit risk assessment
- Probability-based risk banding (low/medium/high)
- Batch prediction support
- Health check endpoint for monitoring

## API Documentation

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `POST /predict`
Make a credit risk prediction.

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
- `medium`: 0.15 ≤ probability ≤ 0.35
- `high`: probability > 0.35

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`. API documentation at `http://localhost:8000/docs`.

## Testing

```bash
pytest tests/
```

## Batch Prediction

```bash
python src/batch.py --input data.csv --output predictions.csv
```

## Deployment

For production deployment, use:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```