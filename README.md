# Production ML Pipeline with FastAPI

A production-ready credit scoring prediction API built with FastAPI.

## Overview

This project exposes a trained credit scoring model via a REST API with health checking, single predictions, and batch prediction support.

## API Documentation

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

## Running Locally

```bash
pip install -r requirements.txt
python run_api.py
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Running Tests

```bash
pip install -r requirements.txt pytest httpx
pytest tests/
```

## Deployment

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Or use the production runner:

```bash
python run_api.py
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── run_api.py
├── models/               # Model artifacts (downloaded on first run)
├── src/
│   ├── __init__.py
│   ├── api.py            # FastAPI app
│   ├── model.py          # Model loading
│   ├── predict.py        # Pydantic schemas & prediction logic
│   └── batch.py          # Batch prediction script
└── tests/
    ├── __init__.py
    └── test_api.py
```