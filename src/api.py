"""FastAPI application with credit prediction endpoints."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import credit_model
from .predict import PredictInput, PredictOutput, predict

app = FastAPI(
    title="Credit Scoring API",
    description="Real-time credit default prediction API with risk band classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load model artifacts on startup."""
    credit_model.load()


@app.get(
    "/health",
    summary="Health check",
    response_model=dict,
    responses={200: {"description": "Service is healthy"}},
)
def health():
    """
    Health check endpoint.

    Returns the service status and whether the model was successfully loaded.
    """
    return {"status": "ok", "model_loaded": credit_model.is_loaded}


@app.post(
    "/predict",
    summary="Predict credit approval",
    response_model=PredictOutput,
    responses={
        200: {"description": "Prediction result"},
        400: {"description": "Invalid input"},
        503: {"description": "Model not loaded"},
    },
)
def predict_endpoint(input_data: PredictInput):
    """
    Make a credit approval prediction.

    Accepts applicant features and returns:
    - **approved**: boolean approval decision
    - **default_probability**: predicted probability of default
    - **risk_band**: low | medium | high
    """
    if not credit_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict(input_data)
    return result
