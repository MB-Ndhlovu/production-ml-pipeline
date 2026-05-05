# Production ML Pipeline with FastAPI

A production-ready credit scoring prediction API built with FastAPI.

## Overview

This project provides a REST API for credit default prediction, enabling real-time scoring of credit applications.

## Tech Stack

- **FastAPI** - Web framework
- **scikit-learn** - ML model
- **joblib** - Model serialization
- **pydantic** - Data validation

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python run_api.py
```

The server starts at `http://localhost:8000`.

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Predict credit default probability.

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

### Risk Bands

| Band | Probability Range |
|------|-------------------|
| low | < 0.15 |
| medium | 0.15 - 0.35 |
| high | > 0.35 |

### Approval Threshold

Applications with `default_probability < 0.35` are approved.

## Running Tests

```bash
pytest tests/
```

## Deployment

For production deployment, use uvicorn with workers:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── run_api.py
├── models/           # ML model artifacts
├── src/
│   ├── __init__.py
│   ├── model.py      # Model loading utilities
│   ├── predict.py    # Prediction logic
│   └── api.py        # FastAPI application
└── tests/
    ├── __init__.py
    └── test_api.py
```