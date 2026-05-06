# Production ML Pipeline with FastAPI

A production-ready credit scoring prediction API built with FastAPI.

## Features

- REST API for real-time credit default predictions
- Probability scores and risk band classification
- Health check endpoint for monitoring
- Batch prediction support
- OpenAPI documentation

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
Predict credit default probability.

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

### Risk Bands
- `low`: probability < 0.15
- `medium`: probability 0.15 - 0.35
- `high`: probability > 0.35

### Approval Threshold
Applications with `default_probability < 0.35` are approved.

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Running Tests

```bash
pytest tests/
```

## Batch Prediction

```bash
python src/batch.py --url http://localhost:8000 --input data.csv --output predictions.csv
```

## Deployment

The API can be deployed using uvicorn:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Model Artifacts

Model artifacts are loaded from the `models/` directory:
- `credit_model.pkl` - Trained classifier
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - Ordered feature names