import joblib
import pandas as pd

def test_model_load():
    model = joblib.load("models/final_model.joblib")
    assert model is not None

def test_model_prediction_shape():
    model = joblib.load("models/final_model.joblib")
    sample = pd.DataFrame([{
        "age": 60, "sex": 1, "cp": 2, "trestbps": 140, "chol": 220,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
    }])
    proba = model.predict_proba(sample)
    assert proba.shape == (1, 2)
