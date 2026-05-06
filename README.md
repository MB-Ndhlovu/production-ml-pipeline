# Production ML Pipeline with FastAPI

A production-ready credit scoring API built with FastAPI, featuring real-time predictions, health checks, and batch processing capabilities.

## Features

- **Real-time Predictions**: POST endpoint for instant credit default probability predictions
- **Health Monitoring**: GET endpoint to verify service and model status
- **Risk Banding**: Classifies applications as low, medium, or high risk
- **Batch Processing**: Script for processing multiple predictions via the API
- **Model Artifacts**: Pre-trained credit scoring model with standardScaler

## API Endpoints

### GET /health
Returns service health status and model loading state.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST /predict
Submit credit application features for prediction.

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

### Risk Band Thresholds
- **Low**: probability < 0.15
- **Medium**: 0.15 <= probability <= 0.35
- **High**: probability > 0.35

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

### Local Development
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Production
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Batch Predictions

Process multiple predictions via the API:
```bash
python src/batch.py --url http://localhost:8000 --input data.csv --output results.json
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
production-ml-pipeline/
├── README.md
├── requirements.txt
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   ├── api.py
│   └── batch.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
└── run_api.py
```