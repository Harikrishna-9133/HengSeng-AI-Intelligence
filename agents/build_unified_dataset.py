
from pathlib import Path
import pandas as pd
import numpy as np

PRICE_FILE = Path("data/processed/hkex_price_features.csv")
LIHKG_FILE = Path("data/processed/lihkg_scored_sentiment.csv")
TELEGRAM_FILE = Path("data/processed/telegram_headline_sentiment.csv")
EXA_FILE = Path("data/processed/exa_headline_sentiment.csv")

OUTPUT_FILE = Path("data/processed/unified_sentiment_price_dataset.csv")

HK_TZ = "Asia/Hong_Kong"
MARKET_CLOSE_HOUR = 16


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def sentiment_to_score(value):
    if pd.isna(value):
        return np.nan

    value = str(value).lower().strip()

    if value in ["positive", "label_2"]:
        return 1
    if value in ["negative", "label_0"]:
        return -1
    if value in ["neutral", "label_1"]:
        return 0

    return np.nan


def prepare_events(df, source, timestamp_col, ticker_col,
                   sentiment_col, confidence_col):

    required = [timestamp_col, ticker_col, sentiment_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{source}: missing column {col}")

    result = pd.DataFrame()

    result["ticker"] = df[ticker_col].astype(str)
    result["timestamp"] = pd.to_datetime(
        df[timestamp_col], utc=True, errors="coerce"
    )

    result["sentiment"] = df[sentiment_col].apply(
        sentiment_to_score
    )

    if confidence_col in df.columns:
        result["confidence"] = pd.to_numeric(
            df[confidence_col], errors="coerce"
        )
    else:
        result["confidence"] = np.nan

    result["source"] = source

    result = result.dropna(
        subset=["timestamp", "ticker", "sentiment"]
    )

    result["timestamp_hk"] = (
        result["timestamp"].dt.tz_convert(HK_TZ)
    )

    return result


def assign_trading_sessions(events, trading_dates):
    """
    Assign each event to the first eligible trading session.

    Before 16:00 HKT: same trading date is eligible.
    At or after 16:00 HKT: next trading date is used.
    Non-trading dates: next available trading date.
    """

    sessions = []

    for ticker, group in events.groupby("ticker"):
        ticker_dates = (
            trading_dates.loc[
                trading_dates["ticker"] == ticker, "Date"
            ]
            .dropna()
            .sort_values()
            .drop_duplicates()
        )

        dates = pd.to_datetime(ticker_dates).dt.normalize().tolist()

        if not dates:
            continue

        for _, event in group.iterrows():
            local_time = event["timestamp_hk"]
            event_date = local_time.tz_localize(None).normalize()

            if local_time.hour >= MARKET_CLOSE_HOUR:
                eligible = [
                    d for d in dates if d > event_date
                ]
            else:
                eligible = [
                    d for d in dates if d >= event_date
                ]

            if not eligible:
                continue

            row = event.to_dict()
            row["session_date"] = eligible[0]
            sessions.append(row)

    return pd.DataFrame(sessions)


def main():
    prices = load_csv(PRICE_FILE)
    lihkg = load_csv(LIHKG_FILE)
    telegram = load_csv(TELEGRAM_FILE)
    exa = load_csv(EXA_FILE)

    prices["Date"] = pd.to_datetime(
        prices["Date"], errors="coerce"
    ).dt.normalize()

    # LIHKG: use individual model sentiment, not the
    # repeated thread-level financial review suggestion.
    lihkg_events = prepare_events(
        lihkg,
        "LIHKG",
        "created_at",
        "ticker",
        "model_sentiment",
        "sentiment_confidence"
    )

    telegram_events = prepare_events(
        telegram,
        "Telegram",
        "published_at",
        "ticker",
        "model_sentiment",
        "confidence"
    )

    exa_events = prepare_events(
        exa,
        "Exa",
        "published_date",
        "ticker",
        "headline_sentiment",
        "confidence"
    )

    events = pd.concat(
        [lihkg_events, telegram_events, exa_events],
        ignore_index=True
    )

    events = assign_trading_sessions(events, prices)

    if events.empty:
        raise ValueError(
            "No sentiment events could be assigned to trading sessions."
        )

    summaries = []

    for (ticker, session), group in events.groupby(
        ["ticker", "session_date"]
    ):
        record = {
            "ticker": ticker,
            "Date": session,
            "total_sentiment_events": len(group),
        }

        for source in ["LIHKG", "Telegram", "Exa"]:
            subset = group[group["source"] == source]

            record[f"{source.lower()}_count"] = len(subset)

            if len(subset):
                record[f"{source.lower()}_mean_sentiment"] = (
                    subset["sentiment"].mean()
                )
                record[f"{source.lower()}_weighted_sentiment"] = (
                    (
                        subset["sentiment"] * subset["confidence"]
                    ).sum()
                    / subset["confidence"].sum()
                    if subset["confidence"].notna().any()
                    and subset["confidence"].sum() > 0
                    else np.nan
                )
            else:
                record[f"{source.lower()}_mean_sentiment"] = np.nan
                record[f"{source.lower()}_weighted_sentiment"] = np.nan

        summaries.append(record)

    sentiment_daily = pd.DataFrame(summaries)

    dataset = prices.merge(
        sentiment_daily,
        on=["ticker", "Date"],
        how="left"
    )

    count_columns = [
        "total_sentiment_events",
        "lihkg_count",
        "telegram_count",
        "exa_count"
    ]

    for col in count_columns:
        dataset[col] = dataset[col].fillna(0).astype(int)

    dataset["has_lihkg"] = (
        dataset["lihkg_count"] > 0
    ).astype(int)

    dataset["has_telegram"] = (
        dataset["telegram_count"] > 0
    ).astype(int)

    dataset["has_exa"] = (
        dataset["exa_count"] > 0
    ).astype(int)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nUNIFIED DATASET CREATED")
    print("Rows:", len(dataset))
    print("Tickers:", dataset["ticker"].nunique())
    print("Date range:",
          dataset["Date"].min(),
          "to",
          dataset["Date"].max())

    print("\nSentiment coverage:")
    print(
        dataset.groupby("ticker")[
            ["lihkg_count", "telegram_count", "exa_count"]
        ].sum().to_string()
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
