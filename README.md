# Production ML Pipeline with FastAPI

Credit scoring prediction API built with FastAPI for serving a scikit-learn model in production.

## Overview

This API serves a pre-trained credit scoring model. It accepts applicant features and returns:
- Approval decision
- Default probability estimate
- Risk band classification

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
Make a credit approval prediction.

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
| Band | Probability Range |
|------|------------------|
| low | < 0.15 |
| medium | 0.15 – 0.35 |
| high | > 0.35 |

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
python run_api.py
```

The server starts at `http://localhost:8000`. API docs available at `/docs`.

## Running Tests

```bash
pytest tests/
```

## Model Artifacts

Model artifacts (`credit_model.pkl`, `scaler.pkl`, `feature_names.pkl`) are downloaded from the [credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository on startup.

## Deployment

Recommended: Uvicorn with a process manager (e.g., systemd, supervisord).

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```