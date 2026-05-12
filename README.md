# Production ML Pipeline with FastAPI

A production-ready credit scoring API built with FastAPI, scikit-learn, and joblib.

## Overview

This API serves predictions from a pre-trained credit scoring model. It accepts applicant features and returns approval decisions, default probabilities, and risk bands.

## Features

- **Real-time predictions** via REST API
- **Batch prediction script** for bulk scoring
- **Health check endpoint** for monitoring
- **OpenAPI documentation** auto-generated
- **Risk stratification** (low/medium/high)

## Installation

```bash
pip install -r requirements.txt
```

## Run locally

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

Submit credit application features for scoring.

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
- `medium`: 0.15 <= probability <= 0.35
- `high`: probability > 0.35

## Batch Predictions

```bash
python src/batch.py --url http://localhost:8000 --input data.csv --output predictions.csv
```

## Run Tests

```bash
pytest tests/ -v
```

## Deployment

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production

Use gunicorn with uvicorn workers:

```bash
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Model Artifacts

Model artifacts (`credit_model.pkl`, `scaler.pkl`, `feature_names.pkl`) are loaded from the `models/` directory. These are sourced from the [credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository.

## License

MIT
