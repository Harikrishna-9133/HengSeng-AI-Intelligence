
import pandas as pd
from pathlib import Path
from transformers import pipeline

INPUT_FILE = Path("data/processed/lihkg_financial_relevant.csv")
OUTPUT_FILE = Path("data/processed/lihkg_sentiment.csv")
SUMMARY_FILE = Path("data/processed/lihkg_sentiment_summary.csv")

# Multilingual sentiment model: initial baseline, not a Hong Kong
# financial-domain model. Validate predictions manually.
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

def find_text_column(df):
    candidates = [
        "original_text",
        "text",
        "content",
        "message",
        "post_text",
        "title"
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"No message-text column found. Available columns: {list(df.columns)}"
    )

def normalize_label(label):
    label = label.lower()

    if "positive" in label or label == "label_2":
        return "positive"
    if "negative" in label or label == "label_0":
        return "negative"
    return "neutral"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The financial relevance dataset is empty.")

    text_column = find_text_column(df)

    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    df = df[df[text_column] != ""].copy()

    print(f"Messages to analyze: {len(df)}")
    print(f"Text column: {text_column}")
    print(f"Loading model: {MODEL_NAME}")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1
    )

    predictions = sentiment_pipeline(
        df[text_column].tolist(),
        batch_size=8,
        truncation=True,
        max_length=256
    )

    df["sentiment"] = [
        normalize_label(result["label"]) for result in predictions
    ]
    df["sentiment_confidence"] = [
        round(float(result["score"]), 4) for result in predictions
    ]

    df["market_direction"] = df["sentiment"].map({
        "positive": "bullish",
        "negative": "bearish",
        "neutral": "neutral"
    })

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    if "ticker" in df.columns:
        summary = (
            df.groupby("ticker")
            .agg(
                message_count=("sentiment", "count"),
                positive=("sentiment", lambda x: (x == "positive").sum()),
                negative=("sentiment", lambda x: (x == "negative").sum()),
                neutral=("sentiment", lambda x: (x == "neutral").sum()),
                average_confidence=("sentiment_confidence", "mean")
            )
            .reset_index()
        )

        summary["net_sentiment"] = (
            summary["positive"] - summary["negative"]
        )

        summary.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nTicker sentiment summary:")
        print(summary.to_string(index=False))

    print(f"\nDetailed results saved: {OUTPUT_FILE}")
    print(f"Summary saved: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
