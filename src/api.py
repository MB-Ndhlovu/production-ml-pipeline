"""FastAPI application for the credit scoring service."""

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.model import load_model
from src.predict import CreditApplication, predict


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0",
)

_model_loaded = False
try:
    load_model()
    _model_loaded = True
except Exception:
    _model_loaded = False


@app.get("/health", response_model=dict)
async def health():
    """
    Health check endpoint.

    Returns the service status and whether the model was successfully loaded.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ok", "model_loaded": _model_loaded},
    )


@app.post("/predict", response_model=dict)
async def credit_predict(application: CreditApplication):
    """
    Submit a credit application for scoring.

    Returns the approval decision, estimated default probability, and risk band.
    """
    result = predict(application)
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )