# Production ML Pipeline

A FastAPI-based microservice for real-time credit default prediction.

## Overview

This service loads a pre-trained credit scoring model and provides a REST API for predicting default probability and risk classification.

## Features

- Real-time single-prediction endpoint
- Batch prediction support
- Health check endpoint
- OpenAPI documentation

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python run_api.py
```

The server starts on `http://localhost:8000`.

## API Endpoints

### GET /health

Health check. Returns model loading status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Predict credit default risk.

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

## Testing

```bash
pytest tests/ -v
```

## Deployment

### Local Production

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── run_api.py
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── model.py
│   ├── predict.py
│   └── batch.py
└── tests/
    ├── __init__.py
    └── test_api.py
```