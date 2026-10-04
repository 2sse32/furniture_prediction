from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    metrics: Dict[str, float]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame,
    target_column: str = "price",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = _build_preprocessor(X)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }

    best_result: TrainingResult | None = None

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }

        current = TrainingResult(model_name=name, pipeline=pipeline, metrics=metrics)
        if best_result is None or current.metrics["rmse"] < best_result.metrics["rmse"]:
            best_result = current

    if best_result is None:
        raise RuntimeError("No model could be trained")

    return best_result


def save_model(pipeline: Pipeline, model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)


def predict(model: Pipeline, records: Iterable[dict]) -> np.ndarray:
    input_df = pd.DataFrame(records)
    return model.predict(input_df)


def train_from_csv(
    csv_path: str | Path,
    target_column: str = "price",
    model_path: str | Path = "model.joblib",
) -> Tuple[str, Dict[str, float]]:
    data = pd.read_csv(csv_path)
    result = train_and_evaluate(data=data, target_column=target_column)
    save_model(result.pipeline, model_path)
    return result.model_name, result.metrics
