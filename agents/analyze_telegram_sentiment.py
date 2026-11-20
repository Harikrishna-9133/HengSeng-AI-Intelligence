import json
import re
from pathlib import Path

import pandas as pd
from transformers import pipeline

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/telegram_sentiment.csv")
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

ASSET_ALIASES = {
    "0700.HK": ["tencent", "騰訊", "腾讯", "0700", "00700"],
    "9988.HK": ["alibaba", "阿里巴巴", "阿里", "9988", "09988"],
    "0005.HK": ["hsbc", "滙豐", "汇丰", "0005", "00005"],
    "1299.HK": ["aia", "友邦", "1299", "01299"],
    "0941.HK": ["china mobile", "中國移動", "中国移动", "0941", "00941"],
}


def latest_input_file():
    files = sorted(RAW_DIR.glob("telegram_posts_*.json"))
    if not files:
        raise FileNotFoundError(
            "No telegram_posts_*.json file found in data/raw. "
            "Run agents/collect_telegram.py first."
        )
    return files[-1]


def find_tickers(text):
    matched = []
    for ticker, aliases in ASSET_ALIASES.items():
        for alias in aliases:
            if alias.isascii() and alias.replace(" ", "").isalnum():
                pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
                found = re.search(pattern, text, re.IGNORECASE)
            else:
                found = alias.lower() in text.lower()
            if found:
                matched.append(ticker)
                break
    return matched


def normalize_label(label):
    label = label.lower()
    if "positive" in label or label == "label_2":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"


def main():
    input_file = latest_input_file()
    with input_file.open("r", encoding="utf-8") as file:
        records = json.load(file)

    posts = [
        post for post in records
        if post.get("type") != "channel" and post.get("text")
    ]
    print("Input file:", input_file)
    print("Individual posts:", len(posts))

    model = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1,
    )
    results = []

    for post in posts:
        text = (post.get("text") or "").strip()
        tickers = find_tickers(text)
        if not tickers:
            continue
        prediction = model(text[:1500], truncation=True, max_length=512)[0]
        sentiment = normalize_label(prediction["label"])
        for ticker in tickers:
            results.append({
                "ticker": ticker,
                "source": "Telegram",
                "channel": post.get("channelTitle"),
                "post_id": post.get("postId"),
                "url": post.get("url"),
                "published_at": post.get("date"),
                "original_text": text,
                "views": post.get("views"),
                "reactions_total": post.get("reactionsTotal"),
                "model_sentiment": sentiment,
                "confidence": round(float(prediction["score"]), 4),
                "sentiment_score": {"positive": 1, "negative": -1, "neutral": 0}[sentiment],
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("\nTELEGRAM SENTIMENT SUMMARY")
    if df.empty:
        print("No target-company mentions found.")
    else:
        summary = df.groupby("ticker").agg(
            matched_posts=("post_id", "nunique"),
            positive=("model_sentiment", lambda x: (x == "positive").sum()),
            negative=("model_sentiment", lambda x: (x == "negative").sum()),
            neutral=("model_sentiment", lambda x: (x == "neutral").sum()),
        ).reset_index()
        print(summary.to_string(index=False))
    print("Total ticker-post records:", len(df))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
