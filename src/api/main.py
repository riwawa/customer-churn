from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional
import time

app = FastAPI(title="Churn Prediction API", version="1.0.0")

artifacts = joblib.load("models/churn_model.pkl")
pipeline = artifacts['pipeline']
feature_cols = artifacts['feature_cols']

class CustomerFeatures(BaseModel):
    tenure: int = Field(0, ge=0)
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(0.0, ge=0)
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"
    InternetService: str = "Fiber optic"
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    PaperlessBilling: str = "Yes"

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    prediction_time_ms: float

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "1.0.0", "metrics": artifacts['metrics']}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    start = time.time()
    try:
        from src.etl.feature_engineering import create_features, encode_features
        df = pd.DataFrame([customer.model_dump()])
        df = create_features(df)
        df = encode_features(df)
        df = df.reindex(columns=feature_cols, fill_value=0)

        proba = pipeline.predict_proba(df)[0, 1]
        risk = "high" if proba > 0.7 else "medium" if proba > 0.4 else "low"

        return PredictionResponse(
            churn_probability=round(float(proba), 4),
            churn_prediction=bool(proba > 0.4),
            risk_level=risk,
            prediction_time_ms=round((time.time() - start) * 1000, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
