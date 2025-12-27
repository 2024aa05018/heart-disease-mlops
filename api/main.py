from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/final_model.joblib"

# Load model at startup
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease risk using a trained ML model",
    version="1.0.0"
)

# Input schema
class PatientInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(input: PatientInput):
    data = pd.DataFrame([{
        "age": input.age,
        "sex": input.sex,
        "cp": input.cp,
        "trestbps": input.trestbps,
        "chol": input.chol,
        "fbs": input.fbs,
        "restecg": input.restecg,
        "thalach": input.thalach,
        "exang": input.exang,
        "oldpeak": input.oldpeak,
        "slope": input.slope,
        "ca": input.ca,
        "thal": input.thal
    }])

    proba = model.predict_proba(data)[0]
    prediction = int(proba[1] >= 0.5)

    return {
        "prediction": prediction,
        "confidence": round(float(proba[1]), 3)
    }

