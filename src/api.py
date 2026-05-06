"""FastAPI application for credit scoring predictions."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import src.model as model_module
from src.predict import PredictionInput, PredictionOutput, predict_default_probability, get_risk_band


app = FastAPI(
    title="Credit Scoring API",
    description="Production ML pipeline for real-time credit default predictions",
    version="1.0.0",
)


@app.get("/health", response_model=dict)
def health_check():
    """
    Health check endpoint.

    Returns the current health status of the service and whether the model
    artifacts are successfully loaded.
    """
    model_loaded = model_module.is_model_loaded()
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": model_loaded,
        }
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predict credit default probability.

    Accepts credit application features and returns the predicted default
    probability along with risk banding.

    - **approved**: True if default probability < 0.5, False otherwise
    - **default_probability**: Predicted probability of default (0-1)
    - **risk_band**: low (<0.15), medium (0.15-0.35), or high (>0.35)
    """
    try:
        probability = predict_default_probability(input_data)
        risk_band = get_risk_band(probability)
        approved = probability < 0.5

        return PredictionOutput(
            approved=approved,
            default_probability=round(probability, 4),
            risk_band=risk_band,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")