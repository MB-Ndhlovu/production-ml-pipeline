# Production ML Pipeline with FastAPI

A production-ready credit scoring API built with FastAPI, exposing a trained model for real-time predictions and batch processing.

## Features

- **Real-time predictions** via REST API
- **Health check endpoint** for monitoring
- **Batch prediction script** for bulk scoring
- **Pydantic validation** for request/response schemas
- **OpenAPI documentation** auto-generated

## API Endpoints

### `POST /predict`

Submit credit application features for scoring.

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

### `GET /health`

Returns API health status and model load state.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Risk Bands

| Band | Probability Range |
|------|-------------------|
| Low | < 0.15 |
| Medium | 0.15 – 0.35 |
| High | > 0.35 |

## Local Development

```bash
pip install -r requirements.txt
python run_api.py
```

API available at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

## Run Tests

```bash
pytest tests/
```

## Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run with gunicorn for production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── run_api.py
├── models/              # Model artifacts
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── model.py        # Artifact loading
│   ├── predict.py      # Prediction logic + schemas
│   ├── api.py          # FastAPI app
│   └── batch.py        # Batch prediction script
└── tests/
    ├── __init__.py
    └── test_api.py
```