from app.config import (
    APIFY_API_TOKEN,
    EXA_API_KEY
)

print("HengSeng Agent Setup Check")
print("-" * 35)

print(
    "Apify credentials:",
    "Configured" if APIFY_API_TOKEN else "Missing"
)

print(
    "Exa credentials:",
    "Configured" if EXA_API_KEY else "Missing"
)

print("Project configuration loaded successfully.")