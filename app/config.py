import os
from dotenv import load_dotenv

load_dotenv()

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
EXA_API_KEY = os.getenv("EXA_API_KEY")

LIHKG_ACTOR_ID = os.getenv("claude_code_reviewer/lihkg-scraper-en")
TELEGRAM_ACTOR_ID = os.getenv("pulselab/telegram-channel-scraper")

ASSETS = {
    "0700.HK": "Tencent Holdings",
    "9988.HK": "Alibaba Group",
    "0005.HK": "HSBC Holdings",
    "1299.HK": "AIA Group",
    "0941.HK": "China Mobile"
}

TIMEZONE = "Asia/Hong_Kong"
LOOKBACK_DAYS = 10

LIHKG_ACTOR_ID = "claude_code_reviewer/lihkg-scraper"
TELEGRAM_ACTOR_ID = "pulselab/telegram-channel-scraper"
TELEGRAM_CHANNELS = [
    "@RakutenTradeIdeas",
    "@phillipcapitalstocks"
]