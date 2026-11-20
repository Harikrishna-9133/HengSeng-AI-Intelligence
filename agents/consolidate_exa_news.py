import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def normalize_title(title):
    return re.sub(r"\s+", " ", (title or "").lower()).strip()


def normalize_url(url):
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = [
        (key, value) for key, value in parse_qsl(parts.query)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"fbclid", "gclid"}
    ]
    return urlunsplit((
        parts.scheme.lower(), parts.netloc.lower(),
        parts.path.rstrip("/"), urlencode(query), ""
    ))


def discover_input_files():
    # Exa collector output files are timestamped. The optional HSBC retry
    # file is included when present.
    files = sorted(RAW_DIR.glob("exa_news_*.json"))
    retry_file = RAW_DIR / "exa_hsbc_retry.json"
    if retry_file.exists():
        files.append(retry_file)
    return files


def main():
    input_files = discover_input_files()
    if not input_files:
        raise FileNotFoundError(
            "No Exa collection JSON files found in data/raw. "
            "Run agents/collect_exa_news.py first."
        )

    articles = []
    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as file:
            records = json.load(file)
        for record in records:
            record["input_file"] = input_file.name
            articles.append(record)

    print("Input files:", len(input_files))
    print("Total input records:", len(articles))
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=10)
    in_window, out_of_window = [], []

    for article in articles:
        published = article.get("published_date")
        try:
            published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
            (in_window if cutoff <= published_dt <= now else out_of_window).append(article)
        except (ValueError, AttributeError):
            article["date_validation"] = "invalid_or_missing"
            out_of_window.append(article)

    unique, duplicates = [], []
    seen_urls, seen_titles = set(), set()
    for article in in_window:
        url_key = normalize_url(article.get("url"))
        title_key = normalize_title(article.get("title"))
        if (url_key and url_key in seen_urls) or (title_key and title_key in seen_titles):
            duplicates.append({
                "duplicate_title": article.get("title"),
                "duplicate_url": article.get("url"),
                "reason": "duplicate_url" if url_key in seen_urls else "duplicate_title",
                "ticker": article.get("ticker"),
            })
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique.append(article)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "exa_news_consolidated.json": unique,
        "exa_news_duplicates.json": duplicates,
        "exa_news_out_of_window.json": out_of_window,
    }
    for filename, records in outputs.items():
        (PROCESSED_DIR / filename).write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print("\nCONSOLIDATION COMPLETE")
    print("Input records:", len(articles))
    print("Within date window:", len(in_window))
    print("Unique articles:", len(unique))
    print("Duplicates:", len(duplicates))
    print("Outside/invalid date records:", len(out_of_window))
    print("\nUnique articles by ticker:")
    for ticker in sorted({a.get("ticker", "Unknown") for a in unique}):
        print(ticker, sum(a.get("ticker", "Unknown") == ticker for a in unique))
    print("\nSaved outputs in:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
