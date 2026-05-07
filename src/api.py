from fastapi import FastAPI, HTTPException
from src.model import model, scaler, feature_names
from src.predict import PredictionInput, PredictionOutput

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    description="Real-time credit default prediction with risk banding",
)


@app.get("/health", response_model=dict)
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict credit default risk for an applicant.

    Returns the probability of default and a risk band.
    """
    try:
        result = __import__("src.predict").predict.predict(
            input_data, model, scaler, feature_names
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))