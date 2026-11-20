
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exa_py import Exa
from app.config import EXA_API_KEY

def main():
    exa = Exa(api_key=EXA_API_KEY)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=10)

    response = exa.search_and_contents(
        "HSBC Holdings 0005 Hong Kong stock financial results "
        "earnings shares banking business",
        type="auto",
        category="news",
        num_results=10,
        start_published_date=start_date.isoformat(),
        end_published_date=end_date.isoformat(),
        text={"max_characters": 3000},
    )

    articles = []

    for result in response.results:
        articles.append({
            "source": "exa",
            "ticker": "0005.HK",
            "company": "HSBC Holdings",
            "title": getattr(result, "title", None),
            "url": getattr(result, "url", None),
            "published_date": getattr(result, "published_date", None),
            "text": getattr(result, "text", None),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        })

    output = Path("data/raw/exa_hsbc_retry.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("HSBC retry completed.")
    print("Articles retrieved:", len(articles))
    print("Saved:", output)

if __name__ == "__main__":
    main()
