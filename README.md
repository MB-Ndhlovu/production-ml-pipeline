# Production ML Pipeline with FastAPI

A production-ready credit scoring prediction API built with FastAPI, scikit-learn, and joblib.

## Overview

This project provides a RESTful API for credit default prediction, with model artifact loading from a trained pipeline.

## Features

- **Credit Risk Prediction**: Predict default probability and risk bands
- **Health Check**: Monitor API and model status
- **Batch Prediction**: Process multiple predictions via script
- **OpenAPI Documentation**: Auto-generated API docs at `/docs`

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
python run_api.py
```

API runs at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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

Predict credit default risk.

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

**Approval:** `approved` is `true` when `risk_band` is `low` or `medium`.

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
│   ├── model.py
│   ├── predict.py
│   ├── api.py
│   └── batch.py
└── tests/
    ├── __init__.py
    └── test_api.py
```

## Model Artifacts

Model artifacts are downloaded from the credit-scoring-pipeline repository:
- `credit_model.pkl`: Trained classifier
- `scaler.pkl`: Feature scaler
- `feature_names.pkl`: Feature column names

## Running Tests

```bash
pytest tests/
```

## Deployment

For production deployment, use uvicorn with workers:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## License

MIT