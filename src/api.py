from fastapi import FastAPI, HTTPException
from src.predict import PredictionInput, predict_default

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts loan default probability and risk bands for credit applications.",
    version="1.0.0",
)

_model_loaded = False

try:
    from src.model import load_artifacts

    load_artifacts()
    _model_loaded = True
except Exception:
    pass


@app.get("/health", summary="Health check")
def health():
    """Returns API and model status.

    Checks that the API is running and the model artifacts are loaded.
    """
    return {"status": "ok", "model_loaded": _model_loaded}


@app.post("/predict", summary="Predict credit risk")
def predict(input_data: PredictionInput):
    """Predict credit risk for a loan application.

    Accepts JSON with applicant features and returns:
    - **approved**: bool — whether the application is approved
    - **default_probability**: float — predicted probability of default
    - **risk_band**: str — 'low', 'medium', or 'high'

    Risk bands:
    - low: probability < 0.15
    - medium: 0.15 <= probability <= 0.35
    - high: probability > 0.35
    """
    try:
        result = predict_default(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))