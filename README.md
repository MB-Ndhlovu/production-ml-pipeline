# Production ML Pipeline

A FastAPI-powered credit default prediction API with health checks, probabilistic outputs, and risk banding.

## Quick Start

```bash
pip install -r requirements.txt
python run_api.py
```

API runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Endpoints

### `GET /health`
Returns model loading status.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Credit default prediction.

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

Risk bands: `low` (<0.15), `medium` (0.15–0.35), `high` (>0.35)

## Testing

```bash
pytest tests/ -v
```

## Deployment

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```