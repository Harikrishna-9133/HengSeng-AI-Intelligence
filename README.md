# HengSeng Sentiment Price Prediction Agent

## Intern Assessment — CrowdWisdomTrading

A Python-based sentiment and market-analysis pipeline for five Hong Kong-listed assets. It combines retail discussion from LIHKG and Telegram with recent news from Exa, aligns timestamped sentiment events to HKEX trading sessions, derives historical-price features, calculates technical reference levels, and produces experimental next-session return estimates.

> **Research limitation:** Social observations in the collected 10-day sample are sparse and concentrated in Alibaba and HSBC. The common test set in the current fair model comparison contains only five observations. Neither Ridge model outperformed the zero-return baseline on MAE and RMSE. Results are exploratory, not evidence of a reliable trading strategy.

## Assets

| Ticker | Company |
|---|---|
| 0700.HK | Tencent Holdings |
| 9988.HK | Alibaba Group |
| 0005.HK | HSBC Holdings |
| 1299.HK | AIA Group |
| 0941.HK | China Mobile |

## Data sources

- **LIHKG** — Hong Kong retail discussion, collected with Apify actor `claude_code_reviewer/lihkg-scraper`.
- **Telegram** — configured public channels, collected with Apify actor `pulselab/telegram-channel-scraper`.
- **Exa** — recent company and asset-related news.
- **yfinance** — daily historical OHLCV price data used for technical features and next-session target construction.

LIHKG and Telegram are the two social sources selected for the Hong Kong exchange scope; Exa is the complementary news source.

## Pipeline

1. Collect LIHKG and Telegram source data through Apify.
2. Collect recent company news through Exa.
3. Normalize available timestamps and sentiment fields, then assign eligible events to Hong Kong trading sessions.
4. Download daily historical prices and derive lagged returns, moving averages, volatility, volume features, and RSI.
5. Calculate 20-session rolling support/resistance reference levels and previous-session pivot levels.
6. Train price-only and price-plus-sentiment Ridge baselines; evaluate chronologically against a zero-return baseline.
7. Review the results in the Streamlit dashboard.

## Repository structure

```text
agents/          Collection, normalization, feature, evaluation, and prediction scripts
app/             Configuration, setup check, and Streamlit dashboard
models/          Exported prediction and evaluation CSV reports
data/raw/        Locally collected source data (not committed)
data/processed/  Locally generated analysis datasets (not committed)
reports/         Assessment results and limitations
```

## Setup

Python 3.11 is recommended.

1. Clone the repository:

   ```powershell
   git clone https://github.com/Harikrishna-9133/HengSeng-AI-Intelligence.git
   cd HengSeng-AI-Intelligence
   ```

2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create the local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Add your own Apify and Exa credentials to `.env`. Do not commit `.env` or send API tokens in plain-text email.

6. Generate the local datasets and reports using the pipeline notes below, then start the dashboard:

   ```powershell
   streamlit run app/dashboard.py
   ```

The raw and processed datasets are intentionally excluded from GitHub. The dashboard will show missing-data states until local collection and processing have generated the expected files.

## Configuration

`.env.example` includes:

- `APIFY_API_TOKEN`
- `EXA_API_KEY`
- `LIHKG_ACTOR_ID`
- `TELEGRAM_ACTOR_ID`

The actor IDs default to the assessment actors shown in the Data sources section and can be overridden in `.env`. Actor input schemas are actor-specific; confirm each actor's current input schema before rerunning a scrape.

## Reproducing the assessment run

Run commands from the repository root. The collection and processing steps are separate scripts rather than a single unattended orchestration command.

### Prices

```powershell
python agents/collect_price_data.py
python agents/build_price_features.py
```

### LIHKG

```powershell
python agents/collect_lihkg.py
```

`agents/collect_lihkg_details.py` enriches search results using thread IDs. Before running it for a new period, replace the assessment-run `THREAD_IDS` list in that script with the relevant IDs returned by the fresh LIHKG search. Then run:

```powershell
python agents/collect_lihkg_details.py
python agents/normalize_lihkg.py
python agents/filter_financial_threads.py
python agents/analyze_lihkg_sentiment.py
python agents/prepare_financial_review.py
python agents/score_financial_sentiment.py
```

The financial relevance and thread-level review suggestions are manually curated for the assessment sample. They are not ground-truth labels and should be reviewed again for a new collection period.

### Telegram

```powershell
python agents/collect_telegram.py
python agents/analyze_telegram_sentiment.py
python agents/refine_telegram_headlines.py
```

The configured channels are `@RakutenTradeIdeas` and `@phillipcapitalstocks`. Coverage depends on the available public-channel posts and the actor's current input schema.

### Exa news

```powershell
python agents/collect_exa_news.py
python agents/retry_hsbc_news.py
python agents/consolidate_exa_news.py
python agents/analyze_exa_headline_sentiment.py
```

`retry_hsbc_news.py` is an optional retry step for the assessment's HSBC news search.

### Merge, evaluate, predict, and visualize

```powershell
python agents/build_unified_dataset.py
python agents/validate_unified_dataset.py
python agents/calculate_support_resistance.py
python agents/evaluate_models_fairly.py
python agents/predict_next_session.py
streamlit run app/dashboard.py
```

## Evaluation and known limitations

- The prediction target is the next available trading session's return, based on daily closing prices.
- The fair evaluation uses a chronological split, with price-only and price-plus-sentiment Ridge models evaluated on the same sentiment-covered test rows and compared with a naive zero-return baseline.
- The recorded common test set contains five observations. In that comparison, the zero-return baseline has lower MAE and RMSE than both Ridge variants. Directional accuracy for a constant zero-return prediction is not informative.
- Social coverage is uneven across tickers. Missing social observations are not neutral sentiment.
- LIHKG and Telegram source schemas do not consistently provide all requested engagement fields. The assessment run captured available LIHKG likes/dislikes and Telegram views/reactions, but repost counts and author follower counts are not fully populated.
- The dashboard now shows a separate social-only ranking from LIHKG and Telegram, plus a separate combined ranking that includes Exa news. The social-only view indicates observed discussion coverage, not the full market opinion.
- The current pipeline classifies sentiment and derives market direction, but does not reliably extract explicit long/short position statements from every social post.
- XLM-RoBERTa is a general multilingual sentiment model, not a financial-domain model fine-tuned specifically for Hong Kong Cantonese. Automated classifications need human review.
- Support/resistance values are historical reference levels, not guarantees.

See [`reports/ASSESSMENT_RESULTS.md`](reports/ASSESSMENT_RESULTS.md) for the recorded data coverage and model comparison.

## Disclaimer

For research and educational purposes only. This project does not provide investment advice, execution signals, or guarantees of future performance.
