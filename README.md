# Production ML Pipeline

A FastAPI-based credit scoring prediction API that loads pre-trained model artifacts and exposes them via REST endpoints.

## Features

- **Single prediction** via `POST /predict`
- **Health check** via `GET /health`
- **Batch prediction** script via `src/batch.py`
- Auto-loaded scikit-learn model, scaler, and feature names from `models/`

## API Endpoints

### `GET /health`
Returns the health status of the service and whether the model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Accepts a JSON payload with applicant features and returns a credit decision.

**Request:**
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
# Install dependencies
pip install -r requirements.txt

# Run the API server
python run_api.py

# Run tests
pytest
```

## Deployment

The API can be containerized or deployed to any Python host (e.g., Railway, Render, AWS Elastic Beanstalk).

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

## Project Structure

```
production-ml-pipeline/
├── models/           # Model artifact files
├── src/
│   ├── __init__.py
│   ├── api.py        # FastAPI application
│   ├── batch.py      # Batch prediction script
│   ├── model.py      # Artifact loading
│   └── predict.py    # Prediction logic & schemas
├── tests/
│   ├── __init__.py
│   └── test_api.py   # Pytest API tests
├── run_api.py        # Local dev server runner
├── requirements.txt
└── README.md
```