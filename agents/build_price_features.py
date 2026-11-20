
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = Path("data/raw/hkex_historical_prices.csv")
OUTPUT_FILE = Path("data/processed/hkex_price_features.csv")


def calculate_rsi(close, period=14):
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    df = df.sort_values(
        ["ticker", "Date"]
    ).reset_index(drop=True)

    results = []

    for ticker, group in df.groupby("ticker"):
        group = group.copy()
        group = group.sort_values("Date").reset_index(drop=True)

        close = group["Close"]
        volume = group["Volume"]

        # Historical price features
        group["daily_return"] = close.pct_change()

        group["return_lag_1"] = group["daily_return"].shift(1)
        group["return_lag_2"] = group["daily_return"].shift(2)
        group["return_lag_3"] = group["daily_return"].shift(3)

        group["sma_5"] = close.rolling(5).mean()
        group["sma_10"] = close.rolling(10).mean()
        group["sma_20"] = close.rolling(20).mean()

        group["volatility_5"] = (
            group["daily_return"].rolling(5).std()
        )

        group["volatility_10"] = (
            group["daily_return"].rolling(10).std()
        )

        group["average_volume_5"] = volume.rolling(5).mean()
        group["volume_change"] = volume.pct_change()

        group["rsi_14"] = calculate_rsi(close)

        group["price_vs_sma_5"] = (
            close / group["sma_5"] - 1
        )

        group["price_vs_sma_20"] = (
            close / group["sma_20"] - 1
        )

        # Prediction target: next available trading session
        group["next_trading_date"] = group["Date"].shift(-1)

        group["target_next_close"] = close.shift(-1)

        group["target_next_return"] = (
            group["target_next_close"] / close - 1
        )

        results.append(group)

    final_df = pd.concat(results, ignore_index=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPRICE FEATURE DATASET CREATED")
    print("Total records:", len(final_df))

    print("\nRecords by ticker:")
    print(final_df.groupby("ticker").size().to_string())

    print("\nMissing values in selected features:")
    selected = [
        "daily_return",
        "sma_5",
        "sma_20",
        "rsi_14",
        "target_next_close",
        "target_next_return"
    ]

    print(final_df[selected].isna().sum().to_string())

    print("\nSaved:", OUTPUT_FILE)

    print("\nLatest available records:")
    print(
        final_df.groupby("ticker").tail(1)[
            ["ticker", "Date", "Close", "target_next_close"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
