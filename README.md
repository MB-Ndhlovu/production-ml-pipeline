# Production ML Pipeline with FastAPI

A production-grade credit scoring API with model serving, batch prediction, and health monitoring.

## Features

- **Real-time predictions** via REST API
- **Probability output** with risk band classification
- **Batch prediction** script
- **Health checks** for model availability
- **OpenAPI documentation** at `/docs`

## API Endpoints

### `GET /health`
Returns service health and model load status.

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
Accepts credit applicant features and returns a prediction.

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

### Risk Bands

| Band | Probability |
|------|-------------|
| Low | < 0.15 |
| Medium | 0.15 – 0.35 |
| High | > 0.35 |

## Deployment

### Local Development
```bash
pip install -r requirements.txt
python run_api.py
```

### Production
```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Testing
```bash
pytest tests/test_api.py -v
```

## Project Structure

```
production-ml-pipeline/
├── models/           # Model artifacts (.pkl files)
├── src/
│   ├── model.py      # Model loading and prediction logic
│   ├── predict.py    # Pydantic schemas and prediction function
│   ├── api.py        # FastAPI application
│   └── batch.py      # Batch prediction script
├── tests/
│   └── test_api.py   # API tests
├── run_api.py        # Local dev server runner
├── requirements.txt
└── README.md
```