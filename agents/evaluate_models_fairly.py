
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
    "total_sentiment_events",
]

EXCLUDED_COLUMNS = [
    "Date",
    "ticker",
    "target_next_close",
    "target_next_return",
    "next_trading_date",
]


def build_features(train_df, test_df, include_sentiment):

    excluded = EXCLUDED_COLUMNS.copy()

    if not include_sentiment:
        excluded.extend(SENTIMENT_COLUMNS)

    combined = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    numeric_columns = combined.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        column for column in numeric_columns
        if column not in excluded
    ]

    numeric_features = combined[
        feature_columns
    ].copy()

    ticker_dummies = pd.get_dummies(
        combined["ticker"],
        prefix="ticker",
        dtype=float
    )

    X = pd.concat(
        [numeric_features, ticker_dummies],
        axis=1
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.mask(X.abs() > 1e12, np.nan)
    X = X.dropna(axis=1, how="all")

    train_count = len(train_df)

    X_train = X.iloc[:train_count].copy()
    X_test = X.iloc[train_count:].copy()

    valid_columns = X_train.columns[
        X_train.notna().any()
    ]

    X_train = X_train[valid_columns]
    X_test = X_test.reindex(
        columns=valid_columns
    )

    return X_train, X_test


def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Directional accuracy (%)": (
            np.mean(
                np.sign(actual) == np.sign(predicted)
            ) * 100
        ),
    }


def main():

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"]
    )

    df = df.sort_values(
        ["Date", "ticker"]
    ).reset_index(drop=True)

    df = df.dropna(
        subset=[TARGET]
    ).copy()

    # Use only observations with actual sentiment
    # to define a common test period.
    sentiment_data = df[
        df["total_sentiment_events"] > 0
    ].copy()

    unique_dates = sorted(
        sentiment_data["Date"].unique()
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "Not enough sentiment dates for chronological evaluation."
        )

    split_index = int(len(unique_dates) * 0.8)
    split_index = max(
        1,
        min(split_index, len(unique_dates) - 1)
    )

    test_start = unique_dates[split_index]

    # Both models are evaluated on exactly these rows.
    test = sentiment_data[
        sentiment_data["Date"] >= test_start
    ].copy()

    # Price-only model can use all available price history
    # before the common test period.
    price_train = df[
        df["Date"] < test_start
    ].copy()

    # Sentiment model trains only on rows with sentiment.
    sentiment_train = sentiment_data[
        sentiment_data["Date"] < test_start
    ].copy()

    if test.empty or price_train.empty or sentiment_train.empty:
        raise ValueError(
            "Insufficient training or testing observations."
        )

    print("COMMON TEST PERIOD")
    print("Test starts:", test_start)
    print("Test ends:", test["Date"].max())
    print("Common test observations:", len(test))
    print("Common test dates:", test["Date"].nunique())
    print("Price training observations:", len(price_train))
    print("Sentiment training observations:", len(sentiment_train))

    results = []
    prediction_frames = []

    model_configs = [
        (
            "Price-only Ridge",
            price_train,
            False,
        ),
        (
            "Price + Sentiment Ridge",
            sentiment_train,
            True,
        ),
    ]

    for model_name, training, use_sentiment in model_configs:

        X_train, X_test = build_features(
            training,
            test,
            use_sentiment
        )

        y_train = training[TARGET]
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

        results.append({
            "Model": model_name,
            "Training observations": len(training),
            "Test observations": len(test),
            **metrics,
        })

        frame = test[
            ["Date", "ticker", TARGET]
        ].copy()

        frame["predicted_next_return"] = predictions
        frame["model"] = model_name

        prediction_frames.append(frame)

    # Naive baseline: predict zero return for every stock.
    naive_predictions = np.zeros(len(test))

    naive_metrics = calculate_metrics(
        test[TARGET].to_numpy(),
        naive_predictions
    )

    results.append({
        "Model": "Naive Zero-Return Baseline",
        "Training observations": 0,
        "Test observations": len(test),
        **naive_metrics,
    })

    comparison = pd.DataFrame(results)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    comparison_file = (
        OUTPUT_DIR / "fair_model_comparison.csv"
    )

    predictions_file = (
        OUTPUT_DIR / "fair_model_predictions.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    predictions_df.to_csv(
        predictions_file,
        index=False
    )

    print("\nFAIR MODEL COMPARISON")
    print(
        comparison.to_string(index=False)
    )

    print("\nSaved:")
    print(comparison_file)
    print(predictions_file)

    print(
        "\nNote: The common test set is small. "
        "Treat all results as exploratory."
    )


if __name__ == "__main__":
    main()
