# Production ML Pipeline

A FastAPI-powered credit scoring API for real-time and batch inference.

## Overview

This project serves a trained credit scoring model via a REST API. It accepts applicant features and returns a default probability, approval decision, and risk band.

## API Endpoints

### `GET /health`
Health check. Returns model load status.

**Response:**
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
Classify a credit application.

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
  "default_probability": 0.1023,
  "risk_band": "low"
}
```

**Risk bands:**
| Band    | Probability threshold |
|---------|----------------------|
| low     | < 0.15               |
| medium  | 0.15 – 0.35          |
| high    | > 0.35               |

## Setup

```bash
pip install -r requirements.txt
```

## Run locally

```bash
python run_api.py
```

Server starts at `http://localhost:8000`.

## Run tests

```bash
pytest tests/ -v
```

## Batch prediction

```bash
python -m src.batch --input data.csv --url http://localhost:8000
```

## Deployment

Deploy to any platform that supports Python (Railway, Render, Fly.io, etc.):

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

## Model artifacts

Model files are loaded from `models/`:
- `credit_model.pkl` — trained classifier
- `scaler.pkl` — feature scaler
- `feature_names.pkl` — expected feature ordering