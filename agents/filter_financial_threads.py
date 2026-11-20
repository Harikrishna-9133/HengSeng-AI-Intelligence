
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/lihkg_normalized.csv")
OUTPUT_DIR = Path("data/processed")

# Thread-level financial relevance decisions
THREAD_CLASSIFICATION = {
    "4180868": {
        "relevant": True,
        "topic": "Shareholder discussion and share price",
        "reason": "Discussion about HSBC rights issue, dividends and share price."
    },
    "4181761": {
        "relevant": True,
        "topic": "Credit and loan risk",
        "reason": "Discussion about HSBC selling a troubled data-centre loan at a loss."
    },
    "4181848": {
        "relevant": False,
        "topic": "Banking crime incident",
        "reason": "Discussion mainly concerns a suspected fake bank statement."
    },
    "4181954": {
        "relevant": True,
        "topic": "Business strategy",
        "reason": "Discussion about HSBC expansion and Singapore operations."
    },
    "4182420": {
        "relevant": False,
        "topic": "Banking crime incident",
        "reason": "Discussion mainly concerns a suspicious transfer attempt."
    },
    "4179515": {
        "relevant": True,
        "topic": "Share price and AI business",
        "reason": "Discussion about Alibaba share-price movement and AI business."
    },
    "4182148": {
        "relevant": True,
        "topic": "AI chips and export restrictions",
        "reason": "Discussion about Alibaba, NVIDIA and chip restrictions."
    }
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, dtype={"thread_id": str})

    df["thread_id"] = df["thread_id"].astype(str)

    df["financial_relevance"] = df["thread_id"].map(
        lambda thread_id:
            THREAD_CLASSIFICATION.get(
                thread_id, {}
            ).get("relevant", False)
    )

    df["topic"] = df["thread_id"].map(
        lambda thread_id:
            THREAD_CLASSIFICATION.get(
                thread_id, {}
            ).get("topic", "Unclassified")
    )

    df["relevance_reason"] = df["thread_id"].map(
        lambda thread_id:
            THREAD_CLASSIFICATION.get(
                thread_id, {}
            ).get("reason", "No classification available.")
    )

    relevant_df = df[
        df["financial_relevance"] == True
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "lihkg_financial_relevant.csv"

    relevant_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 65)
    print("LIHKG FINANCIAL RELEVANCE FILTER")
    print("=" * 65)

    print("Original rows:", len(df))
    print("Relevant rows:", len(relevant_df))
    print("Excluded rows:", len(df) - len(relevant_df))

    print("\nRelevant threads by ticker:")
    print(
        relevant_df.groupby("ticker")["thread_id"]
        .nunique()
    )

    print("\nRelevant messages by ticker:")
    print(relevant_df.groupby("ticker").size())

    print("\nThread classification:")
    for thread_id, info in THREAD_CLASSIFICATION.items():
        print(
            thread_id,
            "|",
            "KEEP" if info["relevant"] else "EXCLUDE",
            "|",
            info["topic"]
        )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
