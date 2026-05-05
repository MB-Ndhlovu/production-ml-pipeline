# Production ML Pipeline with FastAPI

Credit scoring prediction API powered by a trained scikit-learn model.

## Overview

This project provides a REST API for credit default prediction. It loads pre-trained model artifacts and exposes endpoints for single and batch predictions.

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
Make a credit default prediction.

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

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Testing

```bash
pytest tests/ -v
```

## Deployment

The API can be containerized or deployed to any Python-capable hosting platform:

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Model Artifacts

Model files are loaded from the `models/` directory:
- `credit_model.pkl` – trained classifier
- `scaler.pkl` – feature scaler
- `feature_names.pkl` – expected feature names and categorical mappings