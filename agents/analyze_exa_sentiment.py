
import json
from pathlib import Path

import pandas as pd
from transformers import pipeline

INPUT_FILE = Path("data/processed/exa_news_consolidated.json")
OUTPUT_DIR = Path("data/processed")

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

def normalize_sentiment(label):
    label = label.lower()

    if "positive" in label or label == "label_2":
        return "positive"
    elif "negative" in label or label == "label_0":
        return "negative"
    return "neutral"

def main():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    print("Articles loaded:", len(articles))

    print("Loading multilingual sentiment model...")

    sentiment_model = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )

    records = []

    for index, article in enumerate(articles, start=1):
        title = article.get("title") or ""
        text = article.get("text") or ""

        # Use title and the first part of article text.
        combined_text = f"{title}. {text[:1500]}"

        if not combined_text.strip():
            continue

        try:
            result = sentiment_model(
                combined_text,
                truncation=True,
                max_length=512
            )[0]

            records.append({
                "ticker": article.get("ticker"),
                "company": article.get("company"),
                "title": title,
                "url": article.get("url"),
                "published_date": article.get("published_date"),
                "model_sentiment": normalize_sentiment(
                    result["label"]
                ),
                "confidence": round(
                    float(result["score"]), 4
                ),
                "article_text": text,
                "source": article.get("source", "exa")
            })

            print(
                f"[{index}/{len(articles)}] "
                f"{article.get('ticker')} - "
                f"{normalize_sentiment(result['label'])}"
            )

        except Exception as error:
            print(f"Failed article {index}: {error}")

    df = pd.DataFrame(records)

    detail_file = OUTPUT_DIR / "exa_news_sentiment.csv"
    df.to_csv(detail_file, index=False, encoding="utf-8-sig")

    if not df.empty:
        summary = (
            df.groupby("ticker")
            .agg(
                article_count=("model_sentiment", "count"),
                positive=("model_sentiment",
                          lambda x: (x == "positive").sum()),
                negative=("model_sentiment",
                          lambda x: (x == "negative").sum()),
                neutral=("model_sentiment",
                         lambda x: (x == "neutral").sum()),
                average_confidence=("confidence", "mean")
            )
            .reset_index()
        )

        summary["net_sentiment"] = (
            summary["positive"] - summary["negative"]
        )

        summary_file = (
            OUTPUT_DIR / "exa_news_sentiment_summary.csv"
        )

        summary.to_csv(
            summary_file,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nNEWS SENTIMENT SUMMARY")
        print(summary.to_string(index=False))
        print("\nSummary saved:", summary_file)

    print("\nDetailed results saved:", detail_file)
    print("Articles analyzed:", len(records))

if __name__ == "__main__":
    main()
