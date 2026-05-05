from fastapi import FastAPI
from src.predict import CreditApplication, PredictionResult
from src.model import load_artifacts, is_model_loaded

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production ML pipeline for credit default prediction",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    load_artifacts()


@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current health status of the API and whether
    the model artifacts are loaded.
    """
    return {
        "status": "ok",
        "model_loaded": is_model_loaded()
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(application: CreditApplication):
    """
    Make a credit risk prediction.
    
    Accepts applicant features and returns:
    - approved: whether the application is approved
    - default_probability: predicted probability of default
    - risk_band: low (<0.15), medium (0.15-0.35), or high (>0.35)
    """
    from src.predict import predict_default
    return predict_default(application)