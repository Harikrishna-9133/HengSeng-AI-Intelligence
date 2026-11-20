
import json
from pathlib import Path

files = sorted(Path("data/raw").glob("lihkg_thread_details_*.json"))

if not files:
    raise FileNotFoundError("No LIHKG thread-detail JSON file found.")

latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as f:
    records = json.load(f)

output = []

output.append("LIHKG COMPLETE DATASET INSPECTION")
output.append("=" * 80)
output.append(f"Source file: {latest_file}")
output.append(f"Total records: {len(records)}")

for i, item in enumerate(records, start=1):
    title = item.get("title") or ""
    content = item.get("content") or ""
    own_text = item.get("own_text") or ""
    body = content or own_text

    output.extend([
        "\n" + "=" * 80,
        f"RECORD {i}",
        "=" * 80,
        f"Thread ID: {item.get('thread_id')}",
        f"Title: {title}",
        f"Text length: {len(body)}",
        f"Main text: {body}",
        f"Replies: {item.get('reply_count')}",
        f"Likes: {item.get('like_count')}",
        f"Dislikes: {item.get('dislike_count')}",
        f"Created: {item.get('created_at')}",
        f"URL: {item.get('thread_url')}",
    ])

output_file = Path("data/raw/lihkg_complete_inspection.txt")
output_file.write_text("\n".join(output), encoding="utf-8")

print("Inspection completed.")
print("Total records:", len(records))
print("Saved complete output to:", output_file)
