# Production ML Pipeline with FastAPI

Credit scoring prediction API deployed as a production-ready FastAPI service.

## Overview

REST API for credit default prediction using a trained scikit-learn pipeline. Accepts applicant features and returns approval decision, probability score, and risk band.

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
- `medium`: probability 0.15 – 0.35
- `high`: probability > 0.35

## Local Development

```bash
pip install -r requirements.txt
python run_api.py
```

API available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

## Testing

```bash
pytest tests/
```

## Deployment

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Model Artifacts

Model artifacts (`credit_model.pkl`, `scaler.pkl`, `feature_names.pkl`) are loaded from the [credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository at runtime.