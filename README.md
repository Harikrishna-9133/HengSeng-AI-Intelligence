# HengSeng Sentiment Price Prediction Agent

## Intern Assessment — CrowdWisdomTrading

A Python-based sentiment and market-analysis pipeline for five Hong Kong-listed assets. The pipeline combines retail discussion from LIHKG and Telegram with recent news from Exa, aligns timestamped sentiment events to HKEX trading sessions, creates historical-price features, calculates technical reference levels, and produces experimental next-session return estimates.

> **Research limitation:** Social observations in the collected 10-day sample are sparse and concentrated in Alibaba and HSBC. The shared test set in the current fair model comparison contains only five observations. In that comparison, neither Ridge model outperformed the zero-return baseline on MAE and RMSE. Results are exploratory and are not evidence of a reliable trading strategy.

## Assets

| Ticker | Company |
|---|---|
| 0700.HK | Tencent Holdings |
| 9988.HK | Alibaba Group |
| 0005.HK | HSBC Holdings |
| 1299.HK | AIA Group |
| 0941.HK | China Mobile |

## Data sources

- **LIHKG** — Hong Kong retail discussion, collected with an Apify actor.
- **Telegram** — configured public channels, collected with an Apify actor.
- **Exa** — recent asset-related news articles.
- **yfinance** — daily historical price data used for price features and next-session target construction.

LIHKG and Telegram are the two social sources used for the Hong Kong exchange scope. Exa is the complementary news source.

## Pipeline overview

1. Collect LIHKG and Telegram data through Apify.
2. Collect recent company news with Exa.
3. Normalize source timestamps and sentiment fields; map events to HKEX trading sessions using a Hong Kong market-close cutoff.
4. Download historical daily OHLCV data and derive lagged returns, moving averages, volatility, volume features, and RSI.
5. Calculate 20-session rolling support/resistance reference levels and previous-session pivot levels.
6. Train price-only and price-plus-sentiment Ridge baselines; evaluate chronologically against a zero-return baseline.
7. Explore the outputs in the Streamlit dashboard.

## Repository structure

```text
agents/         Collection, normalization, feature, evaluation, and prediction scripts
app/            Configuration, setup check, and Streamlit dashboard
models/         Exported prediction and evaluation CSV reports
data/raw/       Created locally by collection scripts (not committed)
data/processed/ Created locally by processing scripts (not committed)
```

## Setup

Tested as a local Python project. Python 3.11 is recommended.

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

4. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Set your own credentials in `.env`. Never commit `.env` or share API secrets in plain-text email.

6. Run the dashboard after the local data and report files have been generated:

   ```powershell
   streamlit run app/dashboard.py
   ```

The dashboard reads locally generated files under `data/processed/` and `models/`. Those datasets are intentionally excluded from this repository; therefore the dashboard will show missing-data states until the pipeline has been run.

## Configuration

The local `.env` file uses:

- `APIFY_API_TOKEN`
- `EXA_API_KEY`
- `LIHKG_ACTOR_ID`
- `TELEGRAM_ACTOR_ID`

The Apify actor IDs used for this assessment are configured in `app/config.py`. Actor input schemas and channel lists are actor-specific; confirm the actor's current input schema before rerunning a collection.

## Current evaluation and limitations

- The target is the next available trading session's return, constructed from historical daily closing prices.
- Chronological splitting is used; the evaluation does not randomly shuffle time-series rows.
- Metrics include MAE, RMSE, and directional accuracy.
- The fair comparison evaluates price-only Ridge and price-plus-sentiment Ridge on the same sentiment-covered test rows and includes a naive zero-return baseline.
- The present common test set is extremely small (five observations). The baseline has lower MAE and RMSE than both Ridge variants in the recorded comparison. Directional accuracy for the zero-return baseline is not informative when all predictions are exactly zero.
- Sentiment coverage is uneven across tickers. Missing social observations are kept distinct from neutral sentiment; they should not be interpreted as evidence of neutral investor opinion.
- XLM-RoBERTa is a general multilingual sentiment model, not a financial-domain model fine-tuned specifically for Hong Kong Cantonese discussion. Classifications require human review.
- Support/resistance values are historical technical reference levels, not guarantees.

## Running individual scripts

Scripts are launched from the repository root, for example:

```powershell
python -m app.check_setup
python agents/collect_telegram.py
python agents/collect_lihkg.py
python agents/collect_exa_news.py
python agents/collect_price_data.py
python agents/build_price_features.py
python agents/build_unified_dataset.py
python agents/calculate_support_resistance.py
python agents/evaluate_models_fairly.py
python agents/predict_next_session.py
streamlit run app/dashboard.py
```

Some processing scripts currently refer to timestamped data files from the assessment run. For a fully fresh rerun, update those input paths to the filenames produced by the corresponding latest collection steps, or use the consolidated intermediate files generated in your local run. Do not assume every command above forms a turnkey one-command workflow.

## Disclaimer

For research and educational purposes only. This project does not provide investment advice, execution signals, or guarantees of future performance.
