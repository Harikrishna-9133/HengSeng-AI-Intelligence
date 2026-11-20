
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
    "total_sentiment_events",
]

EXCLUDED_COLUMNS = [
    "Date",
    "ticker",
    "target_next_close",
    "target_next_return",
    "next_trading_date",
]


def build_features(df, include_sentiment):
    excluded = EXCLUDED_COLUMNS.copy()

    if not include_sentiment:
        excluded.extend(SENTIMENT_COLUMNS)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]

    ticker_dummies = pd.get_dummies(
        df["ticker"],
        prefix="ticker",
        dtype=float
    )

    X = pd.concat(
        [
            df[feature_columns].reset_index(drop=True),
            ticker_dummies.reset_index(drop=True),
        ],
        axis=1
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.mask(X.abs() > 1e12, np.nan)
    X = X.dropna(axis=1, how="all")

    return X


def train_and_predict(df, latest, model_name, include_sentiment):

    training = df.dropna(
        subset=[TARGET]
    ).copy()

    prediction_rows = latest.copy()

    if include_sentiment:
        training = training[
            training["total_sentiment_events"] > 0
        ].copy()

        prediction_rows = prediction_rows[
            prediction_rows["total_sentiment_events"] > 0
        ].copy()

    if len(training) < 10:
        print(f"\n{model_name}: Not enough training observations.")
        return None

    if prediction_rows.empty:
        print(f"\n{model_name}: No latest rows with required features.")
        return None

    # Build a common feature matrix so train and prediction
    # columns remain consistent.
    combined = pd.concat(
        [training, prediction_rows],
        ignore_index=True
    )

    X = build_features(
        combined,
        include_sentiment
    )

    train_count = len(training)

    X_train = X.iloc[:train_count].copy()
    X_latest = X.iloc[train_count:].copy()

    # Retain columns with usable training values.
    valid_columns = X_train.columns[
        X_train.notna().any()
    ]

    X_train = X_train[valid_columns]
    X_latest = X_latest.reindex(
        columns=valid_columns
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    y_train = training[TARGET]

    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_latest)

    results = prediction_rows[
        ["Date", "ticker", "company", "Close"]
    ].copy()

    results["model"] = model_name
    results["predicted_next_return"] = predicted_returns

    results["predicted_next_close"] = (
        results["Close"] *
        (1 + results["predicted_next_return"])
    )

    results["predicted_direction"] = np.where(
        results["predicted_next_return"] > 0,
        "UP",
        np.where(
            results["predicted_next_return"] < 0,
            "DOWN",
            "FLAT"
        )
    )

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)
    print("Training observations:", len(training))
    print("Prediction observations:", len(prediction_rows))
    print("Features used:", len(valid_columns))

    print(
        results[
            [
                "ticker",
                "predicted_next_return",
                "predicted_next_close",
                "predicted_direction",
            ]
        ].to_string(index=False)
    )

    return results


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    df = df.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)

    latest_date = df["Date"].max()

    latest = df[
        df["Date"] == latest_date
    ].copy()

    print("LATEST DATA")
    print("Date:", latest_date.date())
    print("Stocks:", len(latest))

    print("\nLatest sentiment coverage:")
    print(
        latest[
            [
                "ticker",
                "lihkg_count",
                "telegram_count",
                "exa_count",
                "total_sentiment_events",
            ]
        ].to_string(index=False)
    )

    price_results = train_and_predict(
        df,
        latest,
        "Price-only Ridge",
        include_sentiment=False
    )

    sentiment_results = train_and_predict(
        df,
        latest,
        "Price + Sentiment Ridge (experimental)",
        include_sentiment=True
    )

    all_results = []

    if price_results is not None:
        all_results.append(price_results)

    if sentiment_results is not None:
        all_results.append(sentiment_results)

    if all_results:
        final_results = pd.concat(
            all_results,
            ignore_index=True
        )

        output_file = (
            OUTPUT_DIR / "next_session_predictions.csv"
        )

        final_results.to_csv(
            output_file,
            index=False
        )

        print("\nPREDICTION REPORT SAVED")
        print(output_file)

    else:
        print("\nNo predictions generated.")


if __name__ == "__main__":
    main()
