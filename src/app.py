import logging
import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure structured logging for API monitoring (Task 8)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("heart_disease_api")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")
model_pipeline = None


def get_model():
    global model_pipeline
    if model_pipeline is None:
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded production model pipeline into memory.")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
    return model_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    get_model()
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Production Inference Service for UCI Heart Disease Risk Prediction",
    version="1.0.0",
    lifespan=lifespan
)


class PatientData(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 63.0}, description="Age in years")
    sex: float = Field(..., json_schema_extra={"example": 1.0}, description="Sex (1 = male, 0 = female)")
    cp: float = Field(..., json_schema_extra={"example": 3.0}, description="Chest pain type (0-3)")
    trestbps: float = Field(..., json_schema_extra={"example": 145.0}, description="Resting blood pressure in mm Hg")
    chol: float = Field(..., json_schema_extra={"example": 233.0}, description="Serum cholesterol in mg/dl")
    fbs: float = Field(..., json_schema_extra={"example": 1.0}, description="Fasting blood sugar > 120 mg/dl (1/0)")
    restecg: float = Field(..., json_schema_extra={"example": 0.0}, description="Resting ECG results (0-2)")
    thalach: float = Field(..., json_schema_extra={"example": 150.0}, description="Maximum heart rate achieved")
    exang: float = Field(..., json_schema_extra={"example": 0.0}, description="Exercise induced angina (1/0)")
    oldpeak: float = Field(..., json_schema_extra={"example": 2.3}, description="ST depression induced by exercise")
    slope: float = Field(..., json_schema_extra={"example": 0.0}, description="Slope of the peak exercise ST segment")
    ca: float = Field(..., json_schema_extra={"example": 0.0}, description="Number of major vessels (0-3)")
    thal: float = Field(..., json_schema_extra={"example": 1.0}, description="Thalassemia status")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Healthy, 1 for Heart Disease")
    prediction_label: str
    confidence: float
    probability_healthy: float
    probability_disease: float


@app.get("/health")
def health_check():
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    data_dict = patient.model_dump()
    logger.info(f"Received inference request: {data_dict}")

    try:
        input_df = pd.DataFrame([data_dict])
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        prob_healthy = float(probabilities[0])
        prob_disease = float(probabilities[1])
        confidence = float(probabilities[prediction])

        label = "Heart Disease Detected" if prediction == 1 else "Healthy / No Disease"

        response = {
            "prediction": prediction,
            "prediction_label": label,
            "confidence": round(confidence, 4),
            "probability_healthy": round(prob_healthy, 4),
            "probability_disease": round(prob_disease, 4),
        }
        logger.info(f"Prediction output: {response}")
        return response
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")
