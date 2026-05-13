# Production ML Pipeline with FastAPI

Credit scoring prediction API built with FastAPI, scikit-learn, and Pydantic.

## Features

- RESTful API for real-time credit default predictions
- Probability scores and risk band classification
- Health check endpoint for monitoring
- Batch prediction support
- OpenAPI documentation

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python run_api.py
```

API available at `http://localhost:8000`

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

**Risk Bands:**
- `low`: probability < 0.15
- `medium`: probability 0.15 - 0.35
- `high`: probability > 0.35

## Batch Predictions

```bash
python -m src.batch --input data.csv --output predictions.csv
```

## Run Tests

```bash
pytest tests/ -v
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_api.py"]
```

### Production

Use `uvicorn src.api:app --host 0.0.0.0 --port 8000` for production deployment.

## Architecture

```
production-ml-pipeline/
├── models/           # ML model artifacts
├── src/              # Source code
│   ├── api.py        # FastAPI application
│   ├── model.py      # Model loading utilities
│   ├── predict.py    # Prediction logic
│   └── batch.py      # Batch processing
├── tests/            # Unit tests
├── run_api.py        # Development server
└── requirements.txt  # Dependencies
```