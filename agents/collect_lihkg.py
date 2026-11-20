import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apify_client import ApifyClient

from app.config import APIFY_API_TOKEN, LIHKG_ACTOR_ID, LOOKBACK_DAYS

KEYWORDS = [
    "Tencent", "腾讯", "0700",
    "Alibaba", "阿里巴巴", "9988",
    "HSBC", "滙豐", "0005",
    "AIA", "友邦", "1299",
    "China Mobile", "中國移動", "0941",
]


def collect_lihkg_posts():
    if not APIFY_API_TOKEN:
        raise ValueError("APIFY_API_TOKEN missing in .env")
    if not LIHKG_ACTOR_ID:
        raise ValueError("LIHKG_ACTOR_ID missing in .env/config")

    date_from = (
        datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    ).strftime("%m/%d/%y")

    client = ApifyClient(APIFY_API_TOKEN)
    run_input = {
        "mode": "search",
        "keywords": KEYWORDS,
        "maxItems": 10,
        "order": "desc_create_time",
        "dateFrom": date_from,
    }

    print("Starting LIHKG collection...")
    print("Keywords:", len(KEYWORDS))
    print(f"Date range: last {LOOKBACK_DAYS} days (from {date_from})")

    run = client.actor(LIHKG_ACTOR_ID).call(run_input=run_input)
    if run is None:
        raise RuntimeError("LIHKG Actor did not return a run")
    dataset_id = run.default_dataset_id
    if not dataset_id:
        raise RuntimeError("No dataset returned by LIHKG Actor")

    items = client.dataset(dataset_id).list_items(clean=True, limit=5000).items
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"lihkg_posts_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2, default=str)

    print("\nLIHKG collection completed.")
    print("Records collected:", len(items))
    print("Actor run ID:", run.id)
    print("Saved to:", output_file)


if __name__ == "__main__":
    collect_lihkg_posts()
