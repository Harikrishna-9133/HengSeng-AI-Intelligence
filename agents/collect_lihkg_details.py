
import json
from datetime import datetime
from pathlib import Path

from apify_client import ApifyClient

from app.config import APIFY_API_TOKEN, LIHKG_ACTOR_ID


# Existing LIHKG thread IDs from the previous collection
THREAD_IDS = [
    "4182148",
    "4180983",
    "4179515",
    "4182420",
    "4181954",
    "4181848",
    "4181761",
    "4180868",
    "4179481",
    "4173194",
]


def main():
    if not APIFY_API_TOKEN:
        raise ValueError("APIFY_API_TOKEN is missing from .env")

    if not LIHKG_ACTOR_ID:
        raise ValueError("LIHKG_ACTOR_ID is missing from config")

    client = ApifyClient(APIFY_API_TOKEN)

    run_input = {
        "mode": "thread",
        "threadIds": THREAD_IDS,
        "maxItems": len(THREAD_IDS),
    }

    print("Starting LIHKG thread-detail collection...")
    print("Threads requested:", len(THREAD_IDS))

    run = client.actor(LIHKG_ACTOR_ID).call(run_input=run_input)

    if not run:
        raise RuntimeError("LIHKG Actor did not return a run")

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("No dataset ID returned by LIHKG Actor")

    records = client.dataset(dataset_id).list_items().items

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"lihkg_thread_details_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    print("\nCollection completed.")
    print("Actor run ID:", run.id)
    print("Records returned:", len(records))
    print("Saved to:", output_file)

    print("\nDETAIL INSPECTION")
    print("=" * 70)

    for index, item in enumerate(records, start=1):
        title = item.get("title") or ""
        content = item.get("content") or ""
        own_text = item.get("own_text") or ""

        print(f"\nRecord {index}")
        print("Thread ID:", item.get("thread_id"))
        print("Title:", title)

        body = content or own_text
        print("Main text length:", len(body))
        print("Main text:", body[:500] if body else "[No main text]")

        # Print likely reply/comment fields, if present
        for field in ("replies", "comments", "messages"):
            value = item.get(field)
            if isinstance(value, list):
                print(f"{field} count:", len(value))
                if value:
                    print(f"First {field[:-1]}:", str(value[0])[:500])

        print("-" * 70)


if __name__ == "__main__":
    main()
