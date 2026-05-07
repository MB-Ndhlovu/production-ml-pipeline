# Production ML Pipeline — Credit Scoring API

REST API for real-time credit default prediction, built with FastAPI.

## Overview

- **Model**: Logistic regression credit default classifier
- **Input**: Applicant features (income, credit score, employment, etc.)
- **Output**: Approval decision, default probability, risk band

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — confirms model is loaded |
| POST | `/predict` | Single prediction with probability + risk band |

### POST /predict

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

### GET /health

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## Risk Bands

| Band | Probability Range |
|------|-------------------|
| low | < 0.15 |
| medium | 0.15 – 0.35 |
| high | > 0.35 |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python run_api.py

# Run tests
pytest tests/
```

## Batch Prediction

```bash
python src/batch.py --input data.csv --output predictions.csv
```

## Deployment

The API is stateless — deploy behind a reverse proxy (nginx, Traefik) and scale horizontally. Use `run_api.py` or uvicorn directly:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```