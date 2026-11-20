
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exa_py import Exa
from app.config import EXA_API_KEY, ASSETS

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_DAYS = 10

def main():
    if not EXA_API_KEY:
        raise ValueError("EXA_API_KEY is not configured.")

    exa = Exa(api_key=EXA_API_KEY)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    all_articles = []

    for ticker, company in ASSETS.items():
        print(f"\nSearching news for {ticker} - {company}")

        query = (
            f"{company} Hong Kong stock financial news "
            f"earnings business market shares"
        )

        try:
            response = exa.search_and_contents(
                query,
                type="auto",
                category="news",
                num_results=10,
                start_published_date=start_date.isoformat(),
                end_published_date=end_date.isoformat(),
                text={"max_characters": 3000},
            )

            for result in response.results:
                article = {
                    "source": "exa",
                    "ticker": ticker,
                    "company": company,
                    "title": getattr(result, "title", None),
                    "url": getattr(result, "url", None),
                    "published_date": getattr(
                        result, "published_date", None
                    ),
                    "text": getattr(result, "text", None),
                    "retrieved_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }

                all_articles.append(article)

            print(f"Articles retrieved: {len(response.results)}")

        except Exception as error:
            print(f"Search failed for {ticker}: {error}")

    output_file = OUTPUT_DIR / (
        f"exa_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            all_articles,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nEXA COLLECTION COMPLETE")
    print("Total articles:", len(all_articles))
    print("Saved:", output_file)

if __name__ == "__main__":
    main()
