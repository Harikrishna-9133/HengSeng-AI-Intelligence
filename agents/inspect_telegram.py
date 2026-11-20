
import json
from pathlib import Path

files = sorted(Path("data/raw").glob("telegram_posts_*.json"))
latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as f:
    records = json.load(f)

posts = [r for r in records if r.get("type") == "post"]

assets = {
    "0700.HK": ["0700", "Tencent"],
    "9988.HK": ["9988", "Alibaba"],
    "0005.HK": ["0005", "HSBC"],
    "1299.HK": ["1299", "AIA Group"],
    "0941.HK": ["0941", "China Mobile"],
}

print("Total posts:", len(posts))

for ticker, keywords in assets.items():
    matched = [
        p for p in posts
        if any(k.lower() in (p.get("text") or "").lower()
               for k in keywords)
    ]

    print(f"\n{ticker}: {len(matched)} mentions")

    for post in matched:
        print("Channel:", post.get("channel"))
        print("Date:", post.get("date"))
        print("Views:", post.get("views"))
        print("Reactions:", post.get("reactions"))
        print("URL:", post.get("url"))
        print("Text:", (post.get("text") or "")[:400])
        print("-" * 50)

if posts:
    print("\nAll fields in a post:")
    print(list(posts[0].keys()))
