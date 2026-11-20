
import json
from pathlib import Path
from transformers import pipeline

INPUT_FILE = Path("data/processed/exa_news_consolidated.json")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    articles = json.load(file)

model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device=-1
)

print("\nHEADLINE-ONLY SENTIMENT TEST\n")

for article in articles[:20]:
    title = article.get("title", "").strip()

    if not title:
        continue

    result = model(title, truncation=True, max_length=128)[0]

    print("-" * 80)
    print("Ticker:", article.get("ticker"))
    print("Headline:", title)
    print("Raw label:", result["label"])
    print("Confidence:", round(result["score"], 4))
