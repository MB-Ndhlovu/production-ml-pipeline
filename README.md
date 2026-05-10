# Production ML Pipeline — Credit Scoring API

FastAPI-powered REST API for real-time and batch credit default prediction.

## Overview

Loads a pre-trained credit scoring model and scaler from the `credit-scoring-pipeline` repo, then exposes:
- Single-prediction endpoint (`POST /predict`)
- Health check endpoint (`GET /health`)

## API Docs

### `GET /health`

Returns API and model status.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Predicts credit default probability for an applicant.

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

**Response**
```json
{
  "approved": true,
  "default_probability": 0.12,
  "risk_band": "low"
}
```

**Risk bands**
| Band   | Probability range |
|--------|-------------------|
| `low`   | `< 0.15`          |
| `medium`| `0.15 – 0.35`    |
| `high`  | `> 0.35`          |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download model artifacts (auto-downloaded on first run)
# Artifacts sourced from: https://github.com/MB-Ndhlovu/credit-scoring-pipeline

# Run the server
python run_api.py
```

## Testing

```bash
pytest tests/ -v
```

## Deployment

Any platform that supports Python + Uvicorn (Render, Railway, Fly.io, Docker):

```bash
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```
