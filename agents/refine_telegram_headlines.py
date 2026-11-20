import json
from pathlib import Path

import pandas as pd
from transformers import pipeline

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/telegram_headline_sentiment.csv")
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

ALIASES = {
    "0700.HK": ["tencent", "騰訊", "腾讯", "0700"],
    "9988.HK": ["alibaba", "阿里巴巴", "阿里", "9988"],
    "0005.HK": ["hsbc", "滙豐", "汇丰", "0005"],
    "1299.HK": ["aia", "友邦", "1299"],
    "0941.HK": ["china mobile", "中國移動", "中国移动", "0941"],
}


def latest_input_file():
    files = sorted(RAW_DIR.glob("telegram_posts_*.json"))
    if not files:
        raise FileNotFoundError(
            "No telegram_posts_*.json file found in data/raw. "
            "Run agents/collect_telegram.py first."
        )
    return files[-1]


def extract_headlines(text):
    rows = []
    for line in (part.strip() for part in text.splitlines() if part.strip()):
        lower_line = line.lower()
        if lower_line.startswith("http") or lower_line.startswith((
            "curated headlines", "south china", "the business", "barron's"
        )):
            continue
        for ticker, aliases in ALIASES.items():
            if any(alias.lower() in lower_line for alias in aliases):
                rows.append({"ticker": ticker, "headline": line})
                break
    return rows


def normalize_label(label):
    label = label.lower()
    if "positive" in label or label == "label_2":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"


def main():
    input_file = latest_input_file()
    with input_file.open("r", encoding="utf-8") as f:
        records = json.load(f)
    posts = [r for r in records if r.get("type") != "channel" and r.get("text")]
    print("Input file:", input_file)
    print("Individual posts:", len(posts))

    model = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)
    rows = []
    for post in posts:
        for item in extract_headlines(post["text"]):
            prediction = model(item["headline"], truncation=True, max_length=128)[0]
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
                "sentiment_score": {"positive": 1, "negative": -1, "neutral": 0}[label],
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("\nREFINED TELEGRAM HEADLINE RESULTS")
    print(df.to_string(index=False) if not df.empty else "No relevant headlines extracted.")
    print("Total extracted headlines:", len(df))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
