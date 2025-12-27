from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import time
import logging
from fastapi import Request

REQUEST_COUNT = 0
TOTAL_LATENCY = 0.0
MODEL_PATH = "models/final_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global REQUEST_COUNT, TOTAL_LATENCY

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT += 1
    TOTAL_LATENCY += duration

    logging.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )
    return response

@app.get("/metrics")
def metrics():
    avg_latency = (
        TOTAL_LATENCY / REQUEST_COUNT if REQUEST_COUNT > 0 else 0
    )
    return {
        "request_count": REQUEST_COUNT,
        "average_latency_seconds": round(avg_latency, 4)
    }

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

