import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import PROCESSED_DIR


def load_data():
    path = PROCESSED_DIR / "heart.csv"
    if not path.exists():
        raise FileNotFoundError("Run preprocess.py first")
    return pd.read_csv(path)


def build_preprocessor(X):
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns.tolist())]
    )


def run_experiment(model, model_name, X, y, params):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        scores = cross_validate(
            model, X, y, cv=5, scoring=scoring, return_train_score=False
        )

        for metric, values in scores.items():
            if metric.startswith("test_"):
                mlflow.log_metric(metric.replace("test_", ""), values.mean())

        # Fit once to log the final model
        model.fit(X, y)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n{model_name} logged to MLflow")


def main():
    mlflow.set_experiment("Heart Disease Classification")

    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    preprocessor = build_preprocessor(X)

    # Logistic Regression
    logreg = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    run_experiment(
        model=logreg,
        model_name="Logistic Regression",
        X=X,
        y=y,
        params={"model": "LogisticRegression", "max_iter": 1000}
    )

    # Random Forest
    rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=42
            ))
        ]
    )

    run_experiment(
        model=rf,
        model_name="Random Forest",
        X=X,
        y=y,
        params={
            "model": "RandomForest",
            "n_estimators": 200,
            "random_state": 42
        }
    )


if __name__ == "__main__":
    main()
