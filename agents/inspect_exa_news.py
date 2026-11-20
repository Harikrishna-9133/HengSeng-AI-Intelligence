
import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/raw/exa_news_20261120_144449.json")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    articles = json.load(file)

print("=" * 70)
print("EXA NEWS DATA INSPECTION")
print("=" * 70)

print("\nTotal articles:", len(articles))

counts = Counter(article.get("ticker") for article in articles)

print("\nArticles by ticker:")
for ticker, count in counts.items():
    print(f"{ticker}: {count}")

print("\n" + "=" * 70)
print("ARTICLE DETAILS")
print("=" * 70)

for index, article in enumerate(articles, start=1):
    print(f"\n[{index}] {article.get('ticker')} - {article.get('company')}")
    print("Title:", article.get("title"))
    print("Published:", article.get("published_date"))
    print("URL:", article.get("url"))
    print("Text available:", bool(article.get("text")))

    text = article.get("text") or ""
    print("Text preview:", text[:350].replace("\n", " "))

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

missing_titles = sum(not a.get("title") for a in articles)
missing_dates = sum(not a.get("published_date") for a in articles)
missing_urls = sum(not a.get("url") for a in articles)
missing_text = sum(not a.get("text") for a in articles)

print("Missing titles:", missing_titles)
print("Missing publication dates:", missing_dates)
print("Missing URLs:", missing_urls)
print("Missing article text:", missing_text)
