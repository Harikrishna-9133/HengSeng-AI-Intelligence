
import json
from pathlib import Path

files = sorted(Path("data/raw").glob("lihkg_posts_*.json"))

if not files:
    raise FileNotFoundError("No LIHKG dataset found")

latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as f:
    records = json.load(f)

print("=" * 80)
print("LIHKG DATASET INSPECTION")
print("=" * 80)

print("File:", latest_file)
print("Total records:", len(records))

for i, item in enumerate(records, start=1):
    title = item.get("title") or ""
    content = item.get("content") or ""
    own_text = item.get("own_text") or ""

    print(f"\n{'=' * 80}")
    print(f"RECORD {i}")
    print("=" * 80)

    print("Thread ID:", item.get("thread_id"))
    print("Title:", title)
    print("Category:", item.get("category"))
    print("Source query:", item.get("source_query"))
    print("Created:", item.get("created_at"))
    print("Last reply:", item.get("last_reply_at"))
    print("Replies:", item.get("reply_count"))
    print("Likes:", item.get("like_count"))
    print("Dislikes:", item.get("dislike_count"))
    print("URL:", item.get("thread_url"))

    print("\nText:")
    print((content or own_text or "[No body text available]")[:1000])

    print("\nAI relevance:", item.get("ai_relevance"))
    print("AI sentiment:", item.get("ai_sentiment_label"))

print("\nInspection completed.")
