
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/lihkg_normalized.csv")

df = pd.read_csv(file_path)

print("=" * 75)
print("NORMALIZED LIHKG THREAD INSPECTION")
print("=" * 75)

print("Total rows:", len(df))
print("Unique threads:", df["thread_id"].nunique())

for ticker in sorted(df["ticker"].dropna().unique()):
    asset_df = df[df["ticker"] == ticker]

    print("\n" + "=" * 75)
    print("TICKER:", ticker)
    print("=" * 75)

    for thread_id, group in asset_df.groupby("thread_id"):
        print("\nThread ID:", thread_id)
        print("Number of rows:", len(group))

        print("\nSample messages:")

        for _, row in group.head(3).iterrows():
            print("-", str(row["original_text"])[:250])

        print("Thread URL:", group["thread_url"].dropna().head(1).tolist())

print("\nInspection completed.")
