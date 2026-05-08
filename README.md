# Production ML Pipeline

A FastAPI-based REST API for credit default prediction, exposing a single `/predict` endpoint and a health check.

## Features

- **REST API** via FastAPI + Uvicorn
- **Single-record prediction** with probability and risk band
- **Batch prediction** via a companion script
- **Health check** endpoint for load-balancer probes
- **OpenAPI documentation** at `/docs`

## Model Artifacts

The pipeline expects three pickled artifacts inside `models/`:

| File | Description |
|------|-------------|
| `credit_model.pkl` | Trained classifier (scikit-learn estimator) |
| `scaler.pkl` | Fitted `StandardScaler` (or equivalent) |
| `feature_names.pkl` | List of feature column names in expected order |

These are downloaded from the
[credit-scoring-pipeline](https://github.com/MB-Ndhlovu/credit-scoring-pipeline) repository.
Replace them with your own trained artifacts as needed.

## API

### `GET /health`

Returns the service health status.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

**Request body**

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

| Field | Type | Description |
|-------|------|-------------|
| `income` | float | Annual income |
| `credit_score` | int | Credit score (300–850 typical) |
| `employment_years` | float | Years employed |
| `debt_to_income` | float | Debt-to-income ratio (0–1) |
| `loan_history_count` | int | Number of past loans |
| `age` | int | Applicant age |
| `home_ownership` | string | One of `"rent"`, `"own"`, `"mortgage"`, `"other"` |
| `verified_income` | int | 1 if income is verified, 0 otherwise |

**Response**

```json
{
  "approved": true,
  "default_probability": 0.12,
  "risk_band": "low"
}
```

**Risk bands**

| Band | Probability range |
|------|--------------------|
| `low` | < 0.15 |
| `medium` | 0.15 – 0.35 |
| `high` | > 0.35 |

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally
python run_api.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs

# Run tests
pytest tests/ -v
```

## Deployment

The API is a standard FastAPI + Uvicorn application. To deploy:

```bash
# Example with gunicorn + uvicorn workers
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Or containerise with Docker:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_api.py"]
```

## Project Structure

```
production-ml-pipeline/
├── models/              # Serialised model artefacts
├── src/
│   ├── __init__.py
│   ├── api.py           # FastAPI app definition
│   ├── model.py         # Model loading utilities
│   ├── predict.py       # Pydantic schemas + prediction logic
│   └── batch.py         # Batch prediction script
├── tests/
│   ├── __init__.py
│   └── test_api.py      # pytest API tests
├── run_api.py           # Local dev entry point
├── requirements.txt
└── README.md
```