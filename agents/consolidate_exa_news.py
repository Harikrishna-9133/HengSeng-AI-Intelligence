
import json
import re
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    RAW_DIR / "exa_news_20261120_144449.json",
    RAW_DIR / "exa_hsbc_retry.json",
]

def normalize_title(title):
    if not title:
        return ""
    return re.sub(r"\s+", " ", title.lower()).strip()

def normalize_url(url):
    if not url:
        return ""

    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"fbclid", "gclid"}
    ]

    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/"),
        urlencode(query),
        ""
    ))

def main():
    articles = []

    for input_file in INPUT_FILES:
        if not input_file.exists():
            print("Missing file:", input_file)
            continue

        with input_file.open("r", encoding="utf-8") as file:
            records = json.load(file)

        for record in records:
            record["input_file"] = input_file.name
            articles.append(record)

    print("Total input records:", len(articles))

    # Filter publication dates to the latest 10 days.
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=10)

    in_window = []
    out_of_window = []

    for article in articles:
        published = article.get("published_date")

        try:
            published_dt = datetime.fromisoformat(
                published.replace("Z", "+00:00")
            )

            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(
                    tzinfo=timezone.utc
                )

            if cutoff <= published_dt <= now:
                in_window.append(article)
            else:
                out_of_window.append(article)

        except (ValueError, AttributeError):
            article["date_validation"] = "invalid_or_missing"
            out_of_window.append(article)

    # Deduplicate by canonical URL, then normalized title.
    unique = []
    duplicates = []
    seen_urls = set()
    seen_titles = set()

    for article in in_window:
        url_key = normalize_url(article.get("url"))
        title_key = normalize_title(article.get("title"))

        if url_key and url_key in seen_urls:
            duplicates.append({
                "duplicate_title": article.get("title"),
                "duplicate_url": article.get("url"),
                "reason": "duplicate_url",
                "ticker": article.get("ticker"),
            })
            continue

        if title_key and title_key in seen_titles:
            duplicates.append({
                "duplicate_title": article.get("title"),
                "duplicate_url": article.get("url"),
                "reason": "duplicate_title",
                "ticker": article.get("ticker"),
            })
            continue

        if url_key:
            seen_urls.add(url_key)

        if title_key:
            seen_titles.add(title_key)

        unique.append(article)

    # Save consolidated dataset.
    output_file = PROCESSED_DIR / "exa_news_consolidated.json"
    output_file.write_text(
        json.dumps(unique, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    duplicate_file = PROCESSED_DIR / "exa_news_duplicates.json"
    duplicate_file.write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    excluded_file = PROCESSED_DIR / "exa_news_out_of_window.json"
    excluded_file.write_text(
        json.dumps(out_of_window, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nCONSOLIDATION COMPLETE")
    print("Input records:", len(articles))
    print("Within date window:", len(in_window))
    print("Unique articles:", len(unique))
    print("Duplicates:", len(duplicates))
    print("Outside/invalid date records:", len(out_of_window))

    print("\nUnique articles by ticker:")
    counts = {}

    for article in unique:
        ticker = article.get("ticker", "Unknown")
        counts[ticker] = counts.get(ticker, 0) + 1

    for ticker, count in sorted(counts.items()):
        print(ticker, count)

    print("\nSaved:")
    print(output_file)
    print(duplicate_file)
    print(excluded_file)

if __name__ == "__main__":
    main()
