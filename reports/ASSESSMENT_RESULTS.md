# HengSeng Sentiment Price Prediction Agent — Assessment Results

## Scope

- Exchange selected: Hong Kong (HKEX)
- Assets: 0700.HK Tencent, 9988.HK Alibaba, 0005.HK HSBC, 1299.HK AIA, 0941.HK China Mobile
- Social sources: LIHKG and Telegram
- News source: Exa
- Price source: yfinance daily adjusted OHLCV history
- Social/news lookback for the collection exercise: 10 days

## Data collected

| Source | Collected volume | Notes |
|---|---:|---|
| LIHKG | 166 thread/reply-level records inspected; 23 normalized ticker rows; 16 retained after financial relevance filtering | Relevant normalized coverage was concentrated in HSBC and Alibaba |
| Telegram | 24 individual posts from the two configured channels | Two Alibaba-related headline records were retained; no meaningful matching coverage for the other four assets in the collected sample |
| Exa | 50 initial records; 49 unique articles after deduplication | 9 HSBC articles and 10 for each of the other four assets |
| Historical prices | 247 daily rows per ticker; 1,235 total | 2025-11-20 to 2026-11-20 in the assessment dataset |

## Sentiment coverage in the merged dataset

| Ticker | LIHKG events | Telegram events | Exa news events |
|---|---:|---:|---:|
| 0005.HK HSBC | 8 | 0 | 9 |
| 0700.HK Tencent | 0 | 0 | 10 |
| 0941.HK China Mobile | 0 | 0 | 10 |
| 1299.HK AIA | 0 | 0 | 10 |
| 9988.HK Alibaba | 8 | 2 | 10 |

Social coverage is therefore very uneven. News provides broad coverage, but LIHKG and Telegram do not provide a representative 10-day social sample for all five stocks. Empty social coverage is not interpreted as neutral sentiment.

## Model evaluation

The fair comparison uses one shared chronological test set of five ticker-session observations, all selected from rows with sentiment coverage. Both Ridge models and the zero-return baseline are evaluated against the same actual target returns.

| Model | Train observations | Test observations | MAE | RMSE | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Price-only Ridge | 1,220 | 5 | 0.025021 | 0.025888 | 40% |
| Price + Sentiment Ridge | 13 | 5 | 0.023824 | 0.029092 | 40% |
| Naive zero-return baseline | 0 | 5 | 0.022983 | 0.024471 | 0%* |

*Directional accuracy for an always-zero prediction is not informative; exact zero predictions are classified as neither a positive nor negative movement by the sign comparison.*

On this tiny sample, the zero-return baseline has the lowest MAE and RMSE. The sentiment model does not demonstrate reliable predictive improvement. The result should be treated as exploratory, not as an investable signal or validated performance claim.

## Next-session estimates generated for the assessment run

The file `models/next_session_predictions.csv` contains two model variants. The price-only variant produced the following estimates from the 2026-11-20 market-data snapshot:

| Ticker | Predicted next return | Predicted next close | Direction |
|---|---:|---:|---|
| 0005.HK HSBC | +0.4275% | 129.7523 | UP |
| 0700.HK Tencent | -0.2922% | 342.9947 | DOWN |
| 0941.HK China Mobile | +0.0786% | 80.1130 | UP |
| 1299.HK AIA | +0.1221% | 74.2906 | UP |
| 9988.HK Alibaba | -0.6973% | 82.7688 | DOWN |

These are model outputs for the assessment snapshot, not verified realized next-session outcomes. They are not investment advice.

## Technical reference levels

Support/resistance levels use the last 20 trading sessions' low/high and prior-session pivot calculations.

| Ticker | Last close | 20-session support | 20-session resistance | Pivot | S1 | R1 |
|---|---:|---:|---:|---:|---:|---:|
| 0005.HK HSBC | 129.20 | 128.60 | 143.30 | 133.10 | 132.50 | 133.90 |
| 0700.HK Tencent | 344.00 | 339.60 | 431.20 | 352.80 | 340.60 | 360.60 |
| 0941.HK China Mobile | 80.05 | 77.00 | 81.70 | 80.60 | 80.20 | 81.10 |
| 1299.HK AIA | 74.20 | 69.80 | 80.00 | 75.80 | 74.90 | 76.45 |
| 9988.HK Alibaba | 83.35 | 80.20 | 109.40 | 81.0833 | 79.6667 | 81.9667 |

These levels are historical technical references. They are not guaranteed market floors or ceilings.

## Known gaps versus the assessment brief

- The collected social sample does not provide meaningful LIHKG/Telegram coverage for all five tickers.
- Available engagement fields vary by actor schema. Telegram views/reactions and LIHKG likes/dislikes were captured where present; author follower counts and repost counts were not fully populated.
- The current pipeline produces sentiment and sentiment-derived direction, but does not robustly extract explicit long/short position statements from posts.
- The dashboard provides a social-only ranking from LIHKG and Telegram as well as a separate combined ranking that includes Exa news.
- The fair model evaluation's common test set is only five observations; a larger, longer, walk-forward evaluation is needed before drawing predictive conclusions.

## Reproducibility and security

Raw/processed datasets are excluded from the public repository. Recreate them locally using the scripts and your own credentials. Apify actor IDs are documented in `README.md` and `app/config.py`. Do not publish APIFY or EXA tokens in a public repository or send them in plain-text email; use a secure secret-sharing method agreed with the recipient.

## Conclusion

The implementation demonstrates a working multi-source ingestion and analysis prototype for the selected Hong Kong assets, with price features, technical levels, a Streamlit report, and a chronological model comparison. However, the social-data sparsity and very small common test set prevent a claim that sentiment improves next-session price prediction. These limitations are intentionally stated to make the results reproducible and interpretable.
