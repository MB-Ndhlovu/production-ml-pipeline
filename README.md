# Production ML Pipeline with FastAPI

A production-ready credit scoring API powered by FastAPI. Serves a trained logistic regression model with standardized predictions, probability outputs, and risk banding.

## Features

- **Real-time predictions** via REST API
- **Probability output** with calibrated risk bands
- **Health check endpoint** for monitoring
- **Batch prediction support** via script
- **OpenAPI documentation** at `/docs`

## API Endpoints

### `GET /health`
Health check. Returns whether the model is loaded and ready.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Submit a credit application for scoring.

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
- `low` — probability < 0.15
- `medium` — probability 0.15 – 0.35
- `high` — probability > 0.35

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python run_api.py
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Deployment

```bash
# Production server (uvicorn)
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── run_api.py
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── model.py       # Model & artifact loading
│   ├── predict.py     # Pydantic schemas & prediction logic
│   ├── api.py         # FastAPI app
│   └── batch.py       # Batch prediction script
└── tests/
    ├── __init__.py
    └── test_api.py
```