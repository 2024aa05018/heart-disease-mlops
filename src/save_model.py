import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DIR

MODEL_PATH = "models/final_model.joblib"


def load_data():
    return pd.read_csv(PROCESSED_DIR / "heart.csv")


def build_pipeline(X):
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns.tolist())]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )
    return pipeline


def main():
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    model = build_pipeline(X)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
