from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import PredictInput, predict

app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction.",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool

    model_config = {"protected_namespaces": ()}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """
    Health check endpoint.

    Confirms the API is running and the model artifacts are loaded.
    """
    from src.model import get_model

    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return HealthResponse(status="ok", is_model_loaded=model_loaded)


@app.post("/predict", response_model=dict, tags=["prediction"])
def predict_endpoint(input_data: PredictInput):
    """
    Run a credit default prediction.

    Accepts applicant features and returns an approval decision,
    default probability, and risk band.

    Risk bands:
      - **low**: probability < 0.15
      - **medium**: 0.15 ≤ probability ≤ 0.35
      - **high**: probability > 0.35
    """
    try:
        result = predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))