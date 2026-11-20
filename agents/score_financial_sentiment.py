
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/lihkg_financial_review.csv")
OUTPUT_FILE = Path("data/processed/lihkg_scored_sentiment.csv")

# Initial transparent review rules.
# These are suggestions, not ground-truth labels.
THREAD_RULES = {
    "4179515": {
        "score": -1,
        "sentiment": "negative",
        "reason": "Share-price weakness and AI competitiveness concerns."
    },
    "4182148": {
        "score": 0,
        "sentiment": "neutral",
        "reason": "Regulatory and reputational implications are uncertain."
    },
    "4180868": {
        "score": 0,
        "sentiment": "neutral",
        "reason": "Mixed shareholder and dividend discussion."
    },
    "4181761": {
        "score": -1,
        "sentiment": "negative",
        "reason": "Reported loss on a troubled loan; stock impact needs validation."
    },
    "4181954": {
        "score": 0,
        "sentiment": "neutral",
        "reason": "Geographic diversification has potentially mixed implications."
    }
}

def main():
    df = pd.read_csv(INPUT_FILE)

    df["financial_sentiment_score"] = pd.NA
    df["financial_sentiment"] = "needs_review"
    df["financial_analysis_reason"] = ""

    for thread_id, rule in THREAD_RULES.items():
        mask = df["thread_id"].astype(str) == thread_id

        df.loc[mask, "financial_sentiment_score"] = rule["score"]
        df.loc[mask, "financial_sentiment"] = rule["sentiment"]
        df.loc[mask, "financial_analysis_reason"] = rule["reason"]

    # Thread-level suggestions are not final individual-message labels.
    df["review_status"] = "pending"
    df["market_direction"] = "needs_review"

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Financial sentiment suggestions generated.")
    print("Total messages:", len(df))

    print("\nSuggested financial sentiment:")
    print(df["financial_sentiment"].value_counts())

    print("\nReview status:")
    print(df["review_status"].value_counts())

    print("\nSaved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
