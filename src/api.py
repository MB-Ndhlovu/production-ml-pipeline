"""FastAPI application for credit risk prediction."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.model import load_artifacts, is_model_loaded, get_model, get_scaler, get_feature_names
from src.predict import PredictionInput, PredictionOutput, RiskBand

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Real-time credit default prediction service",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Load model artifacts on startup."""
    load_artifacts()


@app.get(
    "/health",
    tags=["monitoring"],
    summary="Health check",
    response_model=dict,
)
async def health():
    """
    Returns service health status and model loading state.
    """
    return JSONResponse(
        {"status": "ok", "model_loaded": is_model_loaded()}
    )


@app.post(
    "/predict",
    tags=["prediction"],
    summary="Predict credit default risk",
    response_model=PredictionOutput,
    responses={500: {"description": "Model or prediction error"}},
)
async def predict(input_data: PredictionInput):
    """
    Accept credit application features and return risk assessment.

    - **approved**: true if probability < 0.35, else false
    - **default_probability**: model-predicted probability of default
    - **risk_band**: low (< 0.15), medium (0.15–0.35), high (> 0.35)
    """
    try:
        feature_names = get_feature_names()
        scaler = get_scaler()
        model = get_model()

        feature_values = [
            input_data.income,
            input_data.credit_score,
            input_data.employment_years,
            input_data.debt_to_income,
            input_data.loan_history_count,
            input_data.age,
            input_data.home_ownership,
            input_data.verified_income,
        ]

        scaled = scaler.transform([feature_values])
        probability = float(model.predict_proba(scaled)[0][1])

        if probability < 0.15:
            band = RiskBand.LOW
        elif probability <= 0.35:
            band = RiskBand.MEDIUM
        else:
            band = RiskBand.HIGH

        approved = probability < 0.35

        return PredictionOutput(
            approved=approved,
            default_probability=round(probability, 4),
            risk_band=band,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))