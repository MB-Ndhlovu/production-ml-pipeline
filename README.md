# Production ML Pipeline — Credit Risk Prediction API

FastAPI-based REST API for real-time credit default prediction.

## Features
- Real-time credit risk scoring via REST API
- Probability-based risk banding (low / medium / high)
- Health check endpoint for monitoring
- Batch prediction support via API client
- OpenAPI documentation at `/docs`

## API Endpoints

### `GET /health`
Returns service health and model loading status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`
Accept credit application features and return risk assessment.

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
- `medium`: probability 0.15–0.35
- `high`: probability > 0.35

## Setup

```bash
pip install -r requirements.txt
python run_api.py
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Batch Predictions

```bash
python -m src.batch --api-url http://localhost:8000 --input data.csv --output predictions.csv
```

## Testing

```bash
pytest tests/
```