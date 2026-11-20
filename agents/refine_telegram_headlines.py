
import json
import re
from pathlib import Path

import pandas as pd
from transformers import pipeline

INPUT_FILE = Path("data/raw/telegram_posts_20261120_081455.json")
OUTPUT_FILE = Path("data/processed/telegram_headline_sentiment.csv")

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

ALIASES = {
    "0700.HK": ["tencent", "騰訊", "腾讯"],
    "9988.HK": ["alibaba", "阿里巴巴", "阿里"],
    "0005.HK": ["hsbc", "滙豐", "汇丰"],
    "1299.HK": ["aia", "友邦"],
    "0941.HK": ["china mobile", "中國移動", "中国移动"],
}


def get_ticker(text):
    for ticker, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in text.lower():
                return ticker
    return None


def normalize_label(label):
    label = label.lower()
    if "positive" in label or label == "label_2":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"


def extract_headlines(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    results = []

    for i, line in enumerate(lines):
        ticker = get_ticker(line)

        if not ticker:
            continue

        # Exclude URLs and generic section headings.
        if line.startswith("http") or line.lower().startswith(
            ("curated headlines", "south china", "the business", "barron's")
        ):
            continue

        results.append({
            "ticker": ticker,
            "headline": line
        })

    return results


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        records = json.load(f)

    posts = [
        r for r in records
        if r.get("type") != "channel" and r.get("text")
    ]

    model = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )

    rows = []

    for post in posts:
        headlines = extract_headlines(post["text"])

        for item in headlines:
            prediction = model(
                item["headline"],
                truncation=True,
                max_length=128
            )[0]

            label = normalize_label(prediction["label"])

            rows.append({
                "ticker": item["ticker"],
                "source": "Telegram",
                "post_id": post.get("postId"),
                "url": post.get("url"),
                "published_at": post.get("date"),
                "headline": item["headline"],
                "model_sentiment": label,
                "confidence": round(float(prediction["score"]), 4),
                "sentiment_score": {
                    "positive": 1,
                    "negative": -1,
                    "neutral": 0
                }[label]
            })

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nREFINED TELEGRAM HEADLINE RESULTS")

    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No relevant headlines extracted.")

    print("\nTotal extracted headlines:", len(df))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
