
import json
from pathlib import Path

import pandas as pd


INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

START_DATE = pd.Timestamp("2026-11-10T00:00:00Z")
END_DATE = pd.Timestamp("2026-11-21T00:00:00Z")

ASSET_KEYWORDS = {
    "9988.HK": [
        "阿里巴巴", "阿里", "Alibaba", "9988"
    ],
    "0005.HK": [
        "滙豐", "匯豐", "HSBC", "0005"
    ],
    "0700.HK": [
        "騰訊", "腾讯", "Tencent", "0700"
    ],
    "1299.HK": [
        "友邦", "AIA", "1299"
    ],
    "0941.HK": [
        "中國移動", "中国移动",
        "China Mobile", "0941"
    ],
}


def identify_assets(text):
    matched = []

    text_lower = text.lower()

    for ticker, keywords in ASSET_KEYWORDS.items():
        if any(keyword.lower() in text_lower for keyword in keywords):
            matched.append(ticker)

    return matched


def main():
    files = sorted(
        INPUT_DIR.glob("lihkg_thread_details_*.json")
    )

    if not files:
        raise FileNotFoundError(
            "No LIHKG thread-detail JSON found."
        )

    input_file = files[-1]

    with input_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []

    for item in records:
        text = (
            item.get("content")
            or item.get("own_text")
            or ""
        ).strip()

        created_at = item.get("created_at")

        if not created_at:
            continue

        timestamp = pd.to_datetime(
            created_at,
            utc=True,
            errors="coerce"
        )

        if pd.isna(timestamp):
            continue

        if not (START_DATE <= timestamp < END_DATE):
            continue

        # Combine thread title and text for asset matching.
        full_text = (
            (item.get("title") or "")
            + " "
            + text
        ).strip()

        assets = identify_assets(full_text)

        if not assets or not text:
            continue

        for ticker in assets:
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(rows)

    output_file = (
        OUTPUT_DIR / "lihkg_normalized.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 60)
    print("LIHKG NORMALIZATION COMPLETED")
    print("=" * 60)

    print("Input records:", len(records))
    print("Normalized rows:", len(df))

    if not df.empty:
        print("\nTicker coverage:")
        print(df.groupby("ticker").size())

        print("\nUnique threads:")
        print(df.groupby("ticker")["thread_id"].nunique())

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
