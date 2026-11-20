@'
# HengSeng AI Intelligence

## Agentic Sentiment Analysis & Next-Session Price Prediction

An AI-powered market intelligence platform for Hong Kong-listed stocks (HKEX). It combines social media sentiment, financial news, historical market prices, machine learning predictions, and technical support/resistance analysis.

## Features

- Multi-source sentiment analysis using LIHKG, Telegram, and Exa news.
- Multilingual sentiment analysis using XLM-RoBERTa.
- Historical stock price collection and technical indicators.
- Next-session price prediction using Ridge Regression.
- Price-only and price-plus-sentiment model comparison.
- Support and resistance level calculations.
- Interactive Streamlit dashboard.
- Model evaluation using MAE, RMSE, and directional accuracy.

## Stocks Monitored

| Ticker | Company |
|---|---|
| 0005.HK | HSBC Holdings |
| 0700.HK | Tencent Holdings |
| 0941.HK | China Mobile |
| 1299.HK | AIA Group |
| 9988.HK | Alibaba Group |

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Transformers and PyTorch
- Apify
- Exa
- yfinance
- Streamlit
- Plotly

## Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file using `.env.example` as a reference.
5. Add your own API credentials.
6. Run the dashboard:

   `python -m streamlit run app/dashboard.py`

## Configuration

Configure the following environment variables in your local `.env` file:

- APIFY_API_TOKEN
- EXA_API_KEY
- LIHKG_ACTOR_ID
- TELEGRAM_ACTOR_ID

Never commit API keys or `.env` files.

## Model Evaluation

The project compares a price-only Ridge model, a price-plus-sentiment Ridge model, and a naive zero-return baseline.

The current sentiment dataset has limited coverage. Evaluation results are exploratory and should not be interpreted as evidence of reliable trading performance.

## Disclaimer

This project is developed for research and educational purposes. Predictions are experimental model estimates and are not investment advice.
'@ | Set-Content -Encoding UTF8 README.md