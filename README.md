# Production ML Pipeline — Credit Scoring API

FastAPI-powered REST API for real-time and batch credit risk predictions.

## Overview

Loads a trained credit scoring model (`credit_model.pkl`) with its scaler and feature names from `models/`. Exposes two endpoints:

- `POST /predict` — single prediction with probability and risk band
- `GET /health` — service health check

## API Docs

Once the server is running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

### POST /predict

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

Risk bands:
- `low` — probability < 0.15
- `medium` — probability 0.15 – 0.35
- `high` — probability > 0.35

### GET /health

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python run_api.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Deployment

```bash
# Production server (uvicorn)
uvicorn src.api:app --host 0.0.0.0 --port 8000
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
│   ├── model.py       # artifact loading
│   ├── predict.py     # Pydantic schemas + prediction logic
│   ├── api.py         # FastAPI app
│   └── batch.py       # batch prediction script
└── tests/
    ├── __init__.py
    └── test_api.py
```