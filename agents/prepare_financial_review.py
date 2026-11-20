
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/lihkg_sentiment.csv")
OUTPUT_FILE = Path("data/processed/lihkg_financial_review.csv")

def main():
    df = pd.read_csv(INPUT_FILE)

    # Preserve the original model predictions.
    df["model_sentiment"] = df["sentiment"]
    df["financial_sentiment"] = "needs_review"
    df["event_type"] = "unclassified"
    df["market_direction"] = "needs_review"
    df["analysis_reason"] = ""
    df["review_status"] = "pending"

    # Initial analyst-review suggestions by discussion thread.
    suggestions = {
        "4179515": {
            "event_type": "AI_competition_and_share_price",
            "analysis_reason": (
                "Alibaba share-price weakness and concerns about "
                "AI competitiveness. Review individual comments."
            )
        },
        "4182148": {
            "event_type": "AI_chip_regulation_and_reputation",
            "analysis_reason": (
                "Alibaba-related reporting about NVIDIA, chip access "
                "and possible export-control circumvention."
            )
        },
        "4180868": {
            "event_type": "shareholder_and_dividend_discussion",
            "analysis_reason": (
                "HSBC rights issue, dividend and share-price discussion."
            )
        },
        "4181761": {
            "event_type": "credit_and_loan_risk",
            "analysis_reason": (
                "Reported loss from selling a troubled data-centre loan. "
                "Potentially negative event; overall stock impact needs review."
            )
        },
        "4181954": {
            "event_type": "business_strategy_and_geographic_risk",
            "analysis_reason": (
                "HSBC Singapore expansion and geographic diversification."
            )
        }
    }

    for thread_id, details in suggestions.items():
        mask = df["thread_id"].astype(str) == thread_id
        df.loc[mask, "event_type"] = details["event_type"]
        df.loc[mask, "analysis_reason"] = details["analysis_reason"]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Financial review dataset created.")
    print("Total messages:", len(df))
    print("\nReview status:")
    print(df["review_status"].value_counts())
    print("\nEvent types:")
    print(df["event_type"].value_counts())
    print("\nSaved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
