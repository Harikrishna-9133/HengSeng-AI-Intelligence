
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

ASSETS = {
    "0700.HK": "Tencent Holdings",
    "9988.HK": "Alibaba Group",
    "0005.HK": "HSBC Holdings",
    "1299.HK": "AIA Group",
    "0941.HK": "China Mobile",
}

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    all_prices = []

    for ticker, company in ASSETS.items():
        print(f"\nDownloading {company} ({ticker})...")

        try:
            data = yf.download(
                ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                print(f"No data returned for {ticker}")
                continue

            # Handle both ordinary and MultiIndex columns.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()

            data["ticker"] = ticker
            data["company"] = company

            required = [
                "Date", "Open", "High", "Low",
                "Close", "Volume", "ticker", "company"
            ]

            data = data[required]
            data["Date"] = pd.to_datetime(data["Date"]).dt.date

            all_prices.append(data)

            print("Rows:", len(data))
            print("First date:", data["Date"].min())
            print("Last date:", data["Date"].max())

        except Exception as error:
            print(f"Error downloading {ticker}: {error}")

    if not all_prices:
        raise RuntimeError("No historical price data was downloaded.")

    combined = pd.concat(all_prices, ignore_index=True)

    combined = combined.sort_values(
        ["ticker", "Date"]
    ).reset_index(drop=True)

    output_file = OUTPUT_DIR / "hkex_historical_prices.csv"
    combined.to_csv(output_file, index=False)

    print("\nPRICE COLLECTION SUMMARY")
    print(
        combined.groupby("ticker")
        .agg(
            trading_days=("Date", "count"),
            first_date=("Date", "min"),
            last_date=("Date", "max")
        )
        .to_string()
    )

    print("\nSaved:", output_file)
    print("Total rows:", len(combined))

if __name__ == "__main__":
    main()
