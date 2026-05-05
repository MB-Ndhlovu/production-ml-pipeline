# Production ML Pipeline with FastAPI

Credit risk scoring API for predicting loan default probability and risk bands.

## Overview

This project provides a REST API for credit risk scoring, using a pre-trained model to predict the probability of loan default and assign risk bands.

## Features

- **Predict endpoint**: Classify loan applications as approved/denied with default probability
- **Health check**: Verify API and model status
- **Batch prediction**: Script for processing multiple predictions via the API
- **Risk bands**: Low, Medium, High risk classification based on default probability

## Risk Band Thresholds

| Band | Probability Range |
|------|-------------------|
| Low | < 0.15 |
| Medium | 0.15 – 0.35 |
| High | > 0.35 |

## API Endpoints

### `GET /health`

Returns API and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Predict credit risk for a loan application.

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

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

### Local development:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Production (using uvicorn):
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Running Tests

```bash
pytest tests/ -v
```

## Model Artifacts

The model expects the following artifacts in the `models/` directory:
- `credit_model.pkl` – trained classifier
- `scaler.pkl` – feature scaler
- `feature_names.pkl` – list of feature names in correct order

These are downloaded from the [credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository.

## Deployment

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure model artifacts are in `models/`
4. Run with uvicorn or the provided runner script

For containerized deployment, add a `Dockerfile` with:
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```