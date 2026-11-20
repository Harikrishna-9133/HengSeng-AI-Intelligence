
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = Path(
    "data/processed/unified_sentiment_price_dataset.csv"
)

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "target_next_return"

SENTIMENT_COLUMNS = [
    "lihkg_mean_sentiment",
    "lihkg_weighted_sentiment",
    "telegram_mean_sentiment",
    "telegram_weighted_sentiment",
    "exa_mean_sentiment",
    "exa_weighted_sentiment",
    "lihkg_count",
    "telegram_count",
    "exa_count",
    "has_lihkg",
    "has_telegram",
    "has_exa",
]

EXCLUDED_COLUMNS = [
    "Date",
    "ticker",
    "target_next_close",
    "target_next_return",
    "next_trading_date",
]


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Directional accuracy (%)": directional_accuracy,
    }



def prepare_features(df, include_sentiment):
    excluded = EXCLUDED_COLUMNS.copy()

    if not include_sentiment:
        excluded.extend(SENTIMENT_COLUMNS)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = [
        col for col in numeric_columns
        if col not in excluded
    ]

    ticker_dummies = pd.get_dummies(
        df["ticker"],
        prefix="ticker",
        dtype=float
    )

    X = pd.concat(
        [
            df[features].reset_index(drop=True),
            ticker_dummies.reset_index(drop=True)
        ],
        axis=1
    )

    # Replace infinity and excessively large values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no usable values.
    X = X.dropna(axis=1, how="all")

    # Remove columns containing excessively large values.
    X = X.mask(X.abs() > 1e12, np.nan)

    return X, list(X.columns)



def train_and_evaluate(df, name, include_sentiment):
    # Sentiment model is evaluated only where sentiment exists.
    if include_sentiment:
        df = df[
            df["total_sentiment_events"] > 0
        ].copy()

    df = df.dropna(subset=[TARGET])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 10:
        print(f"\n{name}: Insufficient observations.")
        return None

    unique_dates = sorted(df["Date"].unique())

    split_index = int(len(unique_dates) * 0.8)
    split_index = max(1, min(split_index, len(unique_dates) - 1))

    test_start = unique_dates[split_index]

    train = df[df["Date"] < test_start].copy()
    test = df[df["Date"] >= test_start].copy()

    if len(train) < 5 or len(test) < 1:
        print(f"\n{name}: Insufficient train/test observations.")
        return None

    X, feature_names = prepare_features(
        df,
        include_sentiment
    )

    X_train = X.loc[train.index].copy()
    X_test = X.loc[test.index].copy()

# Replace invalid numeric values.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Remove columns that are entirely missing in training.
    valid_columns = X_train.columns[
        X_train.notna().any()
    ]

    X_train = X_train[valid_columns]
    X_test = X_test.reindex(columns=valid_columns)

# Handle unusually large values.
    X_train = X_train.mask(X_train.abs() > 1e12, np.nan)
    X_test = X_test.mask(X_test.abs() > 1e12, np.nan)

    print("Features used:", len(valid_columns))
    print("Invalid training values:", np.isinf(X_train).sum().sum())
    print("Invalid testing values:", np.isinf(X_test).sum().sum())

    y_train = train[TARGET]
    y_test = test[TARGET]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test.to_numpy(),
        predictions
    )

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print("Observations:", len(df))
    print("Training observations:", len(train))
    print("Testing observations:", len(test))
    print("Training ends:", train["Date"].max())
    print("Testing starts:", test["Date"].min())

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    results = test[
        ["Date", "ticker", TARGET]
    ].copy()

    results["predicted_next_return"] = predictions
    results["model"] = name

    return results, metrics


def main():
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    df = df.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)

    print("DATASET LOADED")
    print("Total rows:", len(df))
    print("Unique dates:", df["Date"].nunique())

    baseline = train_and_evaluate(
        df,
        "Price-only Ridge",
        include_sentiment=False
    )

    sentiment_model = train_and_evaluate(
        df,
        "Price + Sentiment Ridge",
        include_sentiment=True
    )

    if baseline and sentiment_model:
        predictions = pd.concat(
            [baseline[0], sentiment_model[0]],
            ignore_index=True
        )

        predictions.to_csv(
            OUTPUT_DIR / "model_predictions.csv",
            index=False
        )

        print("\nMODEL COMPARISON")
        print(
            pd.DataFrame(
                [
                    {"model": "Price-only Ridge",
                     **baseline[1]},
                    {"model": "Price + Sentiment Ridge",
                     **sentiment_model[1]}
                ]
            ).to_string(index=False)
        )

        print(
            "\nPredictions saved:",
            OUTPUT_DIR / "model_predictions.csv"
        )

    print("\nModel evaluation completed.")
    print(
        "Warning: sentiment evaluation uses a small, "
        "sparse sample. Results are exploratory, not "
        "evidence of reliable trading performance."
    )


if __name__ == "__main__":
    main()
