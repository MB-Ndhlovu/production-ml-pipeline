from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.model import load_artifacts, is_model_loaded
from src.predict import PredictionInput, PredictionOutput, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    try:
        load_artifacts()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=dict,
    summary="Health check",
    description="Returns API status and whether model artifacts are loaded.",
)
def health():
    """
    Health check endpoint.

    Returns:
        dict with status and model_loaded flag.
    """
    return {"status": "ok", "model_loaded": is_model_loaded()}


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict credit default",
    description="Submit applicant features and receive a default probability, approval decision, and risk band.",
)
def predict_endpoint(input_data: PredictionInput):
    """
    Prediction endpoint.

    Args:
        input_data: Validated applicant features.

    Returns:
        PredictionOutput with approval, probability, and risk band.
    """
    return predict(input_data)