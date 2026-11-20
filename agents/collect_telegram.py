
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apify_client import ApifyClient

from app.config import (
    APIFY_API_TOKEN,
    TELEGRAM_ACTOR_ID,
    TELEGRAM_CHANNELS,
)


def collect_telegram_posts():
    if not APIFY_API_TOKEN:
        raise ValueError("APIFY_API_TOKEN is missing in .env")

    if not TELEGRAM_ACTOR_ID:
        raise ValueError("TELEGRAM_ACTOR_ID is missing in config.py")

    if not TELEGRAM_CHANNELS:
        raise ValueError("No Telegram channels configured")

    client = ApifyClient(APIFY_API_TOKEN)

    date_from = (
        datetime.now(timezone.utc).date() - timedelta(days=10)
    ).isoformat()

    run_input = {
        "channels": TELEGRAM_CHANNELS,
        "maxPostsPerChannel": 200,
        "dateFrom": date_from,
        "includeChannelInfo": True,
    }

    print("Starting Telegram collection...")
    print(f"Channels: {TELEGRAM_CHANNELS}")
    print(f"Collecting posts from: {date_from}")

    run = client.actor(TELEGRAM_ACTOR_ID).call(
        run_input=run_input
    )

    if run is None:
        raise RuntimeError("Apify Actor did not return a run result")

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("No dataset ID returned by Apify")

    items = client.dataset(dataset_id).list_items(
        clean=True,
        limit=1000
    ).items

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = output_dir / f"telegram_posts_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            items,
            file,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    posts = [
        item for item in items
        if item.get("type") != "channel"
    ]

    channels = [
        item for item in items
        if item.get("type") == "channel"
    ]

    print("\nCollection completed.")
    print(f"Total records: {len(items)}")
    print(f"Posts collected: {len(posts)}")
    print(f"Channel summaries: {len(channels)}")
    print(f"Saved to: {output_file}")

    for post in posts[:5]:
        print("\n-------------------------")
        print("Channel:", post.get("channel"))
        print("Date:", post.get("date"))
        print("Views:", post.get("views"))
        print("Text:", (post.get("text") or "")[:200])


if __name__ == "__main__":
    collect_telegram_posts()
