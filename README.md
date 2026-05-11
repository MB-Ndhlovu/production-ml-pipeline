# Production ML Pipeline

A FastAPI-powered credit scoring prediction API with health checks, real-time predictions, and batch processing support.

## Overview

This project provides a production-ready ML inference pipeline for credit default prediction. It loads pre-trained model artifacts and exposes them via a documented REST API.

## Features

- **Real-time Predictions** via POST `/predict`
- **Health Checks** via GET `/health`
- **Batch Prediction** script for processing multiple records
- **OpenAPI Documentation** at `/docs`
- **Risk Stratification** with low/medium/high bands

## API Documentation

### GET /health

Returns service health status.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict

Accepts a JSON body with applicant features and returns a credit decision.

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

**Risk Bands:**
- `low`: probability < 0.15
- `medium`: probability 0.15–0.35
- `high`: probability > 0.35

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python run_api.py
```

The server starts on `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

## Running Tests

```bash
pytest tests/
```

## Batch Predictions

```bash
python -m src.batch --input data.csv --output predictions.csv
```

## Deployment

For production deployment, use uvicorn with workers:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Model Artifacts

Model artifacts are loaded from the `models/` directory:
- `credit_model.pkl` — trained classifier
- `scaler.pkl` — feature scaler
- `feature_names.pkl` — expected feature names and ordering