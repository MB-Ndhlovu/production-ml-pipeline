"""FastAPI application for credit scoring."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import is_model_loaded, load_artifacts
from src.predict import CreditApplication, PredictionResult, make_prediction

app = FastAPI(
    title="Credit Scoring API",
    description="Real-time credit default prediction API.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Load model artifacts at startup."""
    load_artifacts()


@app.get("/health", tags=["health"])
async def health():
    """
    Health check endpoint.

    Returns the operational status of the API and whether
    the model artifacts have been loaded successfully.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": is_model_loaded(),
        }
    )


@app.post("/predict", response_model=PredictionResult, tags=["inference"])
async def predict(application: CreditApplication):
    """
    Classify a credit application.

    Accepts applicant features and returns the probability of default,
    an approval decision, and a risk band.

    Risk bands:
    - **low**: probability < 0.15
    - **medium**: 0.15 <= probability <= 0.35
    - **high**: probability > 0.35

    Approval is granted when probability < 0.35.
    """
    try:
        result = make_prediction(application)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))