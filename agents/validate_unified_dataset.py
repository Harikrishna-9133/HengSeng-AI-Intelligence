
import pandas as pd
from pathlib import Path

FILE = Path(
    "data/processed/unified_sentiment_price_dataset.csv"
)

df = pd.read_csv(FILE, parse_dates=["Date"])

print("\n1. DATASET OVERVIEW")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate ticker-date rows:",
      df.duplicated(["ticker", "Date"]).sum())

print("\n2. SENTIMENT COVERAGE")
coverage = df.groupby("ticker").agg(
    trading_days=("Date", "nunique"),
    lihkg_days=("has_lihkg", "sum"),
    telegram_days=("has_telegram", "sum"),
    exa_days=("has_exa", "sum"),
    total_events=("total_sentiment_events", "sum")
)
print(coverage.to_string())

print("\n3. DATES WITH SENTIMENT")
sentiment_rows = df[
    df["total_sentiment_events"] > 0
]

print(
    sentiment_rows[
        ["Date", "ticker", "lihkg_count",
         "telegram_count", "exa_count"]
    ].sort_values(["Date", "ticker"]).to_string(index=False)
)

print("\n4. TARGET VALIDATION")
for col in ["target_next_close", "target_next_return"]:
    if col in df.columns:
        print(col, "missing:", df[col].isna().sum())

print("\n5. SENTIMENT FEATURE MISSING VALUES")
sentiment_cols = [
    c for c in df.columns
    if "sentiment" in c
]

print(df[sentiment_cols].isna().sum().to_string())

print("\n6. RECENT DATA")
print(
    df.sort_values("Date")
    .groupby("ticker")
    .tail(3)[
        ["Date", "ticker", "Close",
         "target_next_close", "target_next_return"]
    ].to_string(index=False)
)

print("\nValidation inspection completed.")
