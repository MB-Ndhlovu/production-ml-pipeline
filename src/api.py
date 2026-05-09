"""FastAPI application with /predict and /health endpoints."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.predict import PredictionInput, PredictionOutput, predict

app = FastAPI(
    title="Credit Scoring API",
    description="Real-time credit risk prediction API powered by FastAPI.",
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=dict,
    tags=["health"],
    summary="Health check",
    responses={200: {"description": "Service is healthy"}},
)
def health():
    """Return the health status of the service.

    The ``model_loaded`` field indicates whether the model artifacts
    were loaded successfully at startup.
    """
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "model_loaded": True},
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["prediction"],
    summary="Submit a credit application for scoring",
    responses={
        200: {"description": "Prediction result with risk band"},
    },
)
def predict_endpoint(input_data: PredictionInput):
    """Score a credit application and return approval, probability, and risk band.

    **Risk bands:**
    - ``low`` — default probability < 0.15
    - ``medium`` — default probability 0.15–0.35
    - ``high`` — default probability > 0.35

    A loan is approved when the probability is below 0.35.
    """
    return predict(input_data)