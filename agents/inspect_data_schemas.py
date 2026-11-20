
import json
from pathlib import Path

import pandas as pd

FILES = [
    Path("data/processed/exa_headline_sentiment.csv"),
    Path("data/processed/lihkg_scored_sentiment.csv"),
    Path("data/raw/telegram_posts_20261120_081455.json"),
    Path("data/raw/hkex_historical_prices.csv"),
]

for file_path in FILES:
    print("\n" + "=" * 80)
    print("FILE:", file_path)

    if not file_path.exists():
        print("NOT FOUND")
        continue

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("\nSample:")
        print(df.head(2).to_string(index=False))

    elif file_path.suffix == ".json":
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        print("Records:", len(data))

        if data:
            print("Fields:", list(data[0].keys()))
            print("\nFirst record:")
            print(json.dumps(data[0], ensure_ascii=False, indent=2)[:2500])
