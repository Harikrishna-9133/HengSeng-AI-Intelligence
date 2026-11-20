
import json
from pathlib import Path

import pandas as pd
from transformers import pipeline

INPUT_FILE = Path("data/processed/exa_news_consolidated.json")
OUTPUT_DIR = Path("data/processed")

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

def normalize_label(label):
    label = label.lower()

    if "positive" in label or label == "label_2":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"

def main():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    print("Articles loaded:", len(articles))

    model = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )

    records = []

    for index, article in enumerate(articles, start=1):
        headline = (article.get("title") or "").strip()

        if not headline:
            continue

        result = model(
            headline,
            truncation=True,
            max_length=128
        )[0]

        sentiment = normalize_label(result["label"])

        # Numerical sentiment: positive = +1, negative = -1.
        score = {
            "positive": 1,
            "negative": -1,
            "neutral": 0
        }[sentiment]

        confidence = float(result["score"])

        records.append({
            "ticker": article.get("ticker"),
            "company": article.get("company"),
            "headline": headline,
            "url": article.get("url"),
            "published_date": article.get("published_date"),
            "headline_sentiment": sentiment,
            "raw_model_label": result["label"],
            "confidence": round(confidence, 4),
            "sentiment_score": score,
            "confidence_weighted_score": round(
                score * confidence, 4
            )
        })

        print(
            f"[{index}/{len(articles)}] "
            f"{article.get('ticker')} | {sentiment} | "
            f"{confidence:.3f}"
        )

    df = pd.DataFrame(records)

    detail_path = OUTPUT_DIR / "exa_headline_sentiment.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    if not df.empty:
        summary = (
            df.groupby("ticker")
            .agg(
                article_count=("headline_sentiment", "count"),
                positive=("headline_sentiment",
                          lambda x: (x == "positive").sum()),
                negative=("headline_sentiment",
                          lambda x: (x == "negative").sum()),
                neutral=("headline_sentiment",
                         lambda x: (x == "neutral").sum()),
                average_confidence=("confidence", "mean"),
                average_weighted_score=(
                    "confidence_weighted_score", "mean"
                )
            )
            .reset_index()
        )

        summary["net_sentiment"] = (
            summary["positive"] - summary["negative"]
        )

        summary_path = (
            OUTPUT_DIR / "exa_headline_sentiment_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nHEADLINE SENTIMENT SUMMARY")
        print(summary.to_string(index=False))
        print("\nSummary saved:", summary_path)

    print("\nDetailed file saved:", detail_path)
    print("Headlines analyzed:", len(records))

if __name__ == "__main__":
    main()
