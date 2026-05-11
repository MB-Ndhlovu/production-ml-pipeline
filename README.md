# Production ML Pipeline with FastAPI

Credit scoring prediction API using a trained scikit-learn model.

## Overview

REST API for credit default prediction with:
- Real-time single prediction via `POST /predict`
- Health check via `GET /health`
- Batch prediction script via `src/batch.py`

## Installation

```bash
pip install -r requirements.txt
```

## Run the API

```bash
python run_api.py
```

The API runs on `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

## API Endpoints

### `GET /health`
Returns service health and model loading status.

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
Accepts credit application features and returns a prediction.

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
- `medium`: probability 0.15 – 0.35
- `high`: probability > 0.35

## Batch Predictions

```bash
python -m src.batch --input data.csv --output predictions.csv --url http://localhost:8000
```

## Deployment

### Local
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Docker
```dockerfile
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Project Structure

```
production-ml-pipeline/
├── models/            # Serialized model artifacts
├── src/
│   ├── model.py       # Model loading utilities
│   ├── predict.py     # Prediction logic & schemas
│   ├── api.py          # FastAPI application
│   └── batch.py        # Batch prediction script
├── tests/
│   └── test_api.py    # API endpoint tests
├── run_api.py         # Local dev server entrypoint
├── requirements.txt
└── README.md
```