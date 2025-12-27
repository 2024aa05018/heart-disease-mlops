import pandas as pd
from src.preprocess import clean_data

def test_clean_data_no_missing():
    raw_df = pd.DataFrame({
        "age": [63, 67],
        "sex": [1, 1],
        "cp": [1, 4],
        "trestbps": [145, 160],
        "chol": [233, 286],
        "fbs": [1, 0],
        "restecg": [2, 2],
        "thalach": [150, 108],
        "exang": [0, 1],
        "oldpeak": [2.3, 1.5],
        "slope": [3, 2],
        "ca": ["0", "3"],      # string on purpose
        "thal": ["6", "3"],    # string on purpose
        "target": [0, 1]
    })

    clean_df = clean_data(raw_df)
    assert clean_df.isnull().sum().sum() == 0


def test_target_binary():
    raw_df = pd.DataFrame({
        "age": [60],
        "sex": [1],
        "cp": [2],
        "trestbps": [140],
        "chol": [220],
        "fbs": [0],
        "restecg": [1],
        "thalach": [150],
        "exang": [0],
        "oldpeak": [1.5],
        "slope": [2],
        "ca": ["0"],
        "thal": ["3"],
        "target": [2]   # >0 should map to 1
    })

    clean_df = clean_data(raw_df)
    assert clean_df["target"].iloc[0] == 1
