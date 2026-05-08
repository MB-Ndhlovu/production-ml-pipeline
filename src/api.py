"""FastAPI application with /predict and /health endpoints."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from src.model import load_artifacts
from src.predict import PredictRequest, PredictResponse, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup; clean up on shutdown."""
    load_artifacts()
    yield


app = FastAPI(
    title="Credit Default Prediction API",
    description="Production ML pipeline for credit default prediction. "
                "Provides single-record prediction with probability and risk band.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    responses={200: {"description": "Service is healthy"}},
)
async def health():
    """
    Returns the service health status and whether the model was loaded.

    Use this endpoint for load-balancer health probes.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "model_loaded": True},
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict credit default",
    responses={
        200: {"description": "Prediction result"},
        422: {"description": "Validation error"},
    },
)
async def predict_endpoint(request: PredictRequest):
    """
    Accept applicant features and return a credit default prediction.

    The response includes:

    - **approved**: whether the application passes the risk threshold
    - **default_probability**: model's probability estimate (0-1)
    - **risk_band**: one of ``low`` (< 0.15), ``medium`` (0.15-0.35), ``high`` (> 0.35)
    """
    return predict(request)