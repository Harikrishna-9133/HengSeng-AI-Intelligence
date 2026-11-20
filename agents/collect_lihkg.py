
import json
from datetime import datetime, timezone
from pathlib import Path

from apify_client import ApifyClient

from app.config import APIFY_API_TOKEN, LIHKG_ACTOR_ID


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
        raise ValueError("LIHKG_ACTOR_ID missing in config.py")

    client = ApifyClient(APIFY_API_TOKEN)

    run_input = {
        "mode": "search",
        "keywords": KEYWORDS,
        "maxItems": 10,
        "order": "desc_create_time",
        "dateFrom": "11/10/26",
    }

    print("Starting LIHKG collection...")
    print("Keywords:", len(KEYWORDS))
    print("Date range: From 10 November 2026")

    run = client.actor(LIHKG_ACTOR_ID).call(
        run_input=run_input
    )

    if run is None:
        raise RuntimeError("LIHKG Actor did not return a run")

    dataset_id = run.default_dataset_id

    if not dataset_id:
        raise RuntimeError("No dataset returned by LIHKG Actor")

    items = client.dataset(dataset_id).list_items(
        clean=True,
        limit=5000
    ).items

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = output_dir / f"lihkg_posts_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            items,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    print("\nLIHKG collection completed.")
    print("Records collected:", len(items))
    print("Saved to:", output_file)

    if items:
        print("\nFirst record fields:")
        print(list(items[0].keys()))

        print("\nFirst record sample:")
        print(
            json.dumps(
                items[0],
                ensure_ascii=False,
                indent=2,
                default=str
            )[:2000]
        )


if __name__ == "__main__":
    collect_lihkg_posts()
