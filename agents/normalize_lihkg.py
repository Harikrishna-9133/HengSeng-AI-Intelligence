import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from app.config import ASSETS, LOOKBACK_DAYS

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

ASSET_KEYWORDS = {
    "9988.HK": ["阿里巴巴", "阿里", "Alibaba", "9988"],
    "0005.HK": ["滙豐", "匯豐", "HSBC", "0005"],
    "0700.HK": ["騰訊", "腾讯", "Tencent", "0700"],
    "1299.HK": ["友邦", "AIA", "1299"],
    "0941.HK": ["中國移動", "中国移动", "China Mobile", "0941"],
}


def identify_assets(text):
    text_lower = text.lower()
    return [
        ticker for ticker, keywords in ASSET_KEYWORDS.items()
        if any(keyword.lower() in text_lower for keyword in keywords)
    ]


def main():
    files = sorted(INPUT_DIR.glob("lihkg_thread_details_*.json"))
    if not files:
        raise FileNotFoundError(
            "No LIHKG thread-detail JSON found in data/raw. "
            "Run the LIHKG detail collection step first."
        )

    input_file = files[-1]
    with input_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - pd.Timedelta(days=LOOKBACK_DAYS)
    rows = []

    for item in records:
        text = (item.get("content") or item.get("own_text") or "").strip()
        created_at = item.get("created_at")
        if not created_at or not text:
            continue

        timestamp = pd.to_datetime(created_at, utc=True, errors="coerce")
        if pd.isna(timestamp) or not (start_date <= timestamp <= end_date):
            continue

        full_text = f"{item.get('title') or ''} {text}".strip()
        for ticker in identify_assets(full_text):
            rows.append({
                "source": "LIHKG",
                "thread_id": item.get("thread_id"),
                "ticker": ticker,
                "original_text": text,
                "language": "zh-Hant",
                "created_at": timestamp.isoformat(),
                "likes": item.get("like_count") or 0,
                "dislikes": item.get("dislike_count") or 0,
                "sentiment_label": None,
                "sentiment_score": None,
                "financial_relevance": None,
                "thread_url": item.get("thread_url"),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "lihkg_normalized.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("LIHKG NORMALIZATION COMPLETED")
    print("Input file:", input_file)
    print("Records inspected:", len(records))
    print("Normalized rows:", len(df))
    if not df.empty:
        print("\nRows by ticker:")
        print(df.groupby("ticker").size().to_string())
        print("\nUnique threads by ticker:")
        print(df.groupby("ticker")["thread_id"].nunique().to_string())
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
