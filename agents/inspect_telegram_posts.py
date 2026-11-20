
import json
from pathlib import Path

file_path = Path("data/raw/telegram_posts_20261120_081455.json")

with file_path.open("r", encoding="utf-8") as file:
    records = json.load(file)

posts = [
    record for record in records
    if record.get("type") != "channel"
]

channels = [
    record for record in records
    if record.get("type") == "channel"
]

print("Total records:", len(records))
print("Channel summaries:", len(channels))
print("Individual posts:", len(posts))

for index, post in enumerate(posts[:5], start=1):
    print("\n" + "=" * 80)
    print("POST", index)

    for field in [
        "inputChannel",
        "channel",
        "channelTitle",
        "postId",
        "url",
        "date",
        "text",
        "views",
        "reactions",
        "reactionsTotal",
    ]:
        print(f"{field}: {post.get(field)}")
