from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "posted_rate"
ID_COLUMN = "load_id"
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
NUMERIC_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", "month", "day_of_week",
    "day_of_month", "day_of_year",
]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date = pd.to_datetime(result.pop("date"), errors="coerce")
    for column in [
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "market_index", "quote_signal",
    ]:
        if column not in result:
            result[column] = np.nan
    result["month"] = date.dt.month
    result["day_of_week"] = date.dt.dayofweek
    result["day_of_month"] = date.dt.day
    result["day_of_year"] = date.dt.dayofyear
    result["distance_per_weight"] = result["distance"] / result["weight"].replace(0, np.nan)
    result["route_distance_lat"] = result["delivery_lat"] - result["pickup_lat"]
    result["route_distance_lon"] = result["delivery_lon"] - result["pickup_lon"]
    return result


def make_pipeline() -> Pipeline:
    numeric_columns = NUMERIC_COLUMNS + [
        "distance_per_weight", "route_distance_lat", "route_distance_lon",
    ]
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ]
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual.to_numpy()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(np.abs(error / actual.to_numpy())) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the freight-rate model and create submission files.")
    parser.add_argument("--train", default="train-test.csv")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--template", default="validation-predictions-template.csv")
    parser.add_argument("--december", default="december-chart-inputs.csv")
    parser.add_argument("--output", default="validation_predictions.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    validation["date"] = pd.to_datetime(validation["date"], errors="coerce")

    train_features = add_features(train.drop(columns=[TARGET, ID_COLUMN]))
    validation_features = add_features(validation.drop(columns=[ID_COLUMN]))

    holdout_mask = train["date"] >= pd.Timestamp("2025-10-01")
    model = make_pipeline()
    model.fit(train_features.loc[~holdout_mask], train.loc[~holdout_mask, TARGET])
    holdout_prediction = model.predict(train_features.loc[holdout_mask])
    print("October holdout metrics:", metrics(train.loc[holdout_mask, TARGET], holdout_prediction))

    model.fit(train_features, train[TARGET])
    validation_prediction = np.maximum(model.predict(validation_features), 0.01)

    template = pd.read_csv(args.template)
    if len(template) != len(validation) or not template[ID_COLUMN].equals(validation[ID_COLUMN]):
        raise ValueError("The prediction template IDs do not match validation.csv in order.")
    output = template.copy()
    output["predicted_rate"] = validation_prediction
    output.to_csv(args.output, index=False)

    december = pd.read_csv(args.december)
    december_features = add_features(december.drop(columns=["predicted_rate"]))
    december["predicted_rate"] = np.maximum(model.predict(december_features), 0.01)
    december.to_csv(args.december, index=False)
    print(f"Wrote {len(output):,} validation predictions to {Path(args.output)}")
    print(f"Wrote {len(december):,} December predictions to {Path(args.december)}")


if __name__ == "__main__":
    main()