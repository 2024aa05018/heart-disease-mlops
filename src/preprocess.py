import pandas as pd
import numpy as np
from src.config import RAW_DIR, PROCESSED_DIR

# Column names as defined by UCI Heart Disease documentation
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

RAW_FILE = RAW_DIR / "processed.cleveland.data"


def load_raw_data() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_FILE}")
    return pd.read_csv(RAW_FILE, header=None, names=COLUMNS)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert target to binary
    # 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)

    return df


def main():
    print("Loading raw data...")
    df = load_raw_data()

    print("Cleaning data...")
    df = clean_data(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "heart.csv"
    df.to_csv(output_path, index=False)

    print("Preprocessing complete ✅")
    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Final dataset shape: {df.shape}")
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
