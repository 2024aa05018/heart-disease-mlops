import pandas as pd
from src.preprocess import load_raw_data, clean_data

def test_clean_data_no_missing():
    df = load_raw_data()
    clean_df = clean_data(df)
    assert clean_df.isnull().sum().sum() == 0

def test_target_binary():
    df = load_raw_data()
    clean_df = clean_data(df)
    assert set(clean_df["target"].unique()).issubset({0, 1})
