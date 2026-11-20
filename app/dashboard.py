
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HengSeng AI | Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

ASSET_NAMES = {
    "0005.HK": "HSBC Holdings",
    "0700.HK": "Tencent Holdings",
    "0941.HK": "China Mobile",
    "1299.HK": "AIA Group",
    "9988.HK": "Alibaba Group",
}

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #929292;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


predictions = load_csv(
    MODELS_DIR / "next_session_predictions.csv"
)

# Load support and resistance levels
support_levels = load_csv(
    Path("data/processed/support_resistance_levels.csv")
)

model_evaluation = load_csv(
    MODELS_DIR / "model_predictions.csv"
)

unified = load_csv(
    PROCESSED_DIR / "unified_sentiment_price_dataset.csv"
)

exa = load_csv(
    PROCESSED_DIR / "exa_headline_sentiment.csv"
)

lihkg = load_csv(
    PROCESSED_DIR / "lihkg_scored_sentiment.csv"
)

telegram = load_csv(
    PROCESSED_DIR / "telegram_headline_sentiment.csv"
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_number(value, default=0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_sentiment(df, candidates):
    """Return a copy with a consistent sentiment column."""

    if df.empty:
        return pd.DataFrame()

    result = df.copy()

    for column in candidates:
        if column in result.columns:
            result["dashboard_sentiment"] = (
                result[column]
                .astype(str)
                .str.lower()
                .str.strip()
            )
            return result

    return pd.DataFrame()


def add_company_names(df):
    if df.empty:
        return df

    result = df.copy()

    if "ticker" in result.columns:
        if "company" not in result.columns:
            result["company"] = result["ticker"].map(
                ASSET_NAMES
            )

        result["company"] = result["company"].fillna(
            result["ticker"].map(ASSET_NAMES)
        )

    return result


def sentiment_counts(df):
    labels = ["positive", "negative", "neutral"]

    if df.empty or "dashboard_sentiment" not in df.columns:
        return pd.DataFrame({
            "Sentiment": labels,
            "Count": [0, 0, 0],
        })

    counts = (
        df["dashboard_sentiment"]
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    return pd.DataFrame({
        "Sentiment": labels,
        "Count": counts.values,
    })


def get_latest_price_data():
    if unified.empty:
        return pd.DataFrame()

    data = unified.copy()

    if "Date" not in data.columns:
        return pd.DataFrame()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.dropna(subset=["Date"])

    latest_date = data["Date"].max()

    return data[data["Date"] == latest_date].copy()


# --------------------------------------------------
# NORMALIZE DATASETS
# --------------------------------------------------

exa = normalize_sentiment(
    exa,
    ["headline_sentiment", "sentiment"]
)

# Use the original model sentiment for LIHKG.
# financial_sentiment contains pending thread-level review labels.
lihkg = normalize_sentiment(
    lihkg,
    ["model_sentiment", "sentiment"]
)

telegram = normalize_sentiment(
    telegram,
    ["model_sentiment", "sentiment"]
)

exa = add_company_names(exa)
lihkg = add_company_names(lihkg)
telegram = add_company_names(telegram)

latest_prices = get_latest_price_data()

# --------------------------------------------------
# UNIFIED SENTIMENT DATA
# --------------------------------------------------

def prepare_source_data(df, source_name):

    if df.empty:
        return pd.DataFrame()

    result = df.copy()

    if "ticker" not in result.columns:
        return pd.DataFrame()

    result["source"] = source_name

    result["dashboard_sentiment"] = (
        result["dashboard_sentiment"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    result = result[
        result["dashboard_sentiment"].isin(
            ["positive", "negative", "neutral"]
        )
    ]

    return result


sentiment_frames = [
    prepare_source_data(lihkg, "LIHKG"),
    prepare_source_data(telegram, "Telegram"),
    prepare_source_data(exa, "Exa News"),
]

sentiment_frames = [
    frame for frame in sentiment_frames
    if not frame.empty
]

all_sentiment = (
    pd.concat(sentiment_frames, ignore_index=True)
    if sentiment_frames
    else pd.DataFrame()
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 Market Intelligence")

st.sidebar.caption(
    "HengSeng Sentiment Price Prediction Agent"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Stock Predictions",
        "Sentiment Analysis",
        "Historical Prices",
        "Model Evaluation",
        "Data Coverage",
    ],
)

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Market: Hong Kong | Exchange: HKEX"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 HengSeng AI Intelligence")

st.markdown(
    "**Agentic Sentiment Analysis & Next-Session "
    "Price Prediction**"
)



if not unified.empty and "Date" in unified.columns:
    dashboard_date = pd.to_datetime(
        unified["Date"],
        errors="coerce"
    ).max()

    st.caption(
        f"Latest available market data: {dashboard_date.date()}"
        if pd.notna(dashboard_date)
        else "Latest market data date unavailable"
    )


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Market Overview")

    if latest_prices.empty:
        st.warning(
            "Unified price dataset is missing or unavailable."
        )
    else:
        total_events = (
            latest_prices["total_sentiment_events"].sum()
            if "total_sentiment_events" in latest_prices
            else 0
        )
        model_count = (
            predictions["model"].nunique()
            if not predictions.empty and "model" in predictions.columns
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Stocks Monitored",
            len(latest_prices)
        )

        col2.metric(
            "Prediction Models",
            int(model_count)
        )

        col3.metric(
            "Latest-Session Sentiment Events",
            int(total_events)
        )

        col4.metric(
            "News Articles",
            len(exa)
        )

    st.divider()

    st.subheader("Latest Stock Predictions")

    if predictions.empty:
        st.warning("Prediction report not found.")
    else:
        display = predictions.copy()

        if "model" in display.columns:
            display = display[
                display["model"] == "Price-only Ridge"
            ]

        display = add_company_names(display)

        columns = [
            "ticker",
            "company",
            "predicted_next_return",
            "predicted_next_close",
            "predicted_direction",
        ]
        columns = [
            column for column in columns
            if column in display.columns
        ]

        # Format the displayed table for interpretation while keeping
        # the original numeric values available for the chart.
        table = display[columns].copy()
        table = table.rename(columns={
            "ticker": "Ticker",
            "company": "Company",
            "predicted_next_return": "Predicted Return",
            "predicted_next_close": "Predicted Next Close",
            "predicted_direction": "Direction",
        })
        if "Predicted Return" in table.columns:
            table["Predicted Return"] = pd.to_numeric(
                table["Predicted Return"], errors="coerce"
            ).map(lambda value: f"{value * 100:+.2f}%" if pd.notna(value) else "N/A")
        if "Predicted Next Close" in table.columns:
            table["Predicted Next Close"] = pd.to_numeric(
                table["Predicted Next Close"], errors="coerce"
            ).map(lambda value: f"{value:,.2f}" if pd.notna(value) else "N/A")

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        if "predicted_next_return" in display.columns:
            chart = display.copy()
            chart["return_percent"] = (
                pd.to_numeric(chart["predicted_next_return"], errors="coerce") * 100
            )

            fig = px.bar(
                chart,
                x="ticker",
                y="return_percent",
                color="predicted_direction",
                title="Predicted Next-Session Returns",
                labels={
                    "return_percent": "Predicted Return (%)",
                    "ticker": "Stock",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("Sentiment Source Overview")

    source_data = pd.DataFrame({
        "Source": ["LIHKG", "Telegram", "Exa News"],
        "Records": [
            len(lihkg),
            len(telegram),
            len(exa),
        ],
    })

    fig = px.bar(
        source_data,
        x="Source",
        y="Records",
        color="Source",
        title="Collected Sentiment Records",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------------------------
# STOCK PREDICTIONS
# --------------------------------------------------


elif page == "Stock Predictions":

    st.subheader("Next-Session Stock Predictions")

    if predictions.empty:
        st.error("Prediction file not found.")

    else:
        available_models = (
            predictions["model"].dropna().unique().tolist()
            if "model" in predictions.columns
            else []
        )

        selected_model = st.selectbox(
            "Select prediction model",
            available_models
        )

        selected = predictions[
            predictions["model"] == selected_model
        ].copy()

        selected = add_company_names(selected)

        # Prediction table
        if "predicted_next_return" in selected.columns:
            selected["Predicted Return (%)"] = (
                selected["predicted_next_return"] * 100
            )

        # Load technical levels
        if not support_levels.empty:

            levels = support_levels.copy()

            selected = selected.merge(
                levels,
                on="ticker",
                how="left",
                suffixes=("", "_levels")
            )

            # Display a compact, readable summary table. Detailed pivot
            # levels remain available in the selected-stock analysis below.
            st.subheader("Stock Prediction & Technical Levels")

            display_columns = [
                "ticker",
                "company",
                "current_price",
                "predicted_next_close",
                "Predicted Return (%)",
                "predicted_direction",
                "support_20d",
                "resistance_20d",
                "price_position",
            ]
            display_columns = [
                col for col in display_columns
                if col in selected.columns
            ]

            table = selected[display_columns].copy()
            table = table.rename(columns={
                "ticker": "Ticker",
                "company": "Company",
                "current_price": "Current Price (HKD)",
                "predicted_next_close": "Predicted Next Close (HKD)",
                "Predicted Return (%)": "Predicted Return",
                "predicted_direction": "Direction",
                "support_20d": "20-Day Support (HKD)",
                "resistance_20d": "20-Day Resistance (HKD)",
                "price_position": "Price Position",
            })

            price_columns = [
                "Current Price (HKD)",
                "Predicted Next Close (HKD)",
                "20-Day Support (HKD)",
                "20-Day Resistance (HKD)",
            ]
            for column in price_columns:
                if column in table.columns:
                    table[column] = pd.to_numeric(
                        table[column], errors="coerce"
                    ).map(
                        lambda value: f"{value:,.2f}" if pd.notna(value) else "N/A"
                    )

            if "Predicted Return" in table.columns:
                table["Predicted Return"] = pd.to_numeric(
                    table["Predicted Return"], errors="coerce"
                ).map(
                    lambda value: f"{value:+.2f}%" if pd.notna(value) else "N/A"
                )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # Select stock for technical analysis
            st.subheader("Individual Stock Support & Resistance")

            selected_ticker = st.selectbox(
                "Choose a stock",
                selected["ticker"].unique().tolist()
            )

            stock = selected[
                selected["ticker"] == selected_ticker
            ].iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current Price",
                f'{stock["current_price"]:.2f}'
                if pd.notna(stock.get("current_price"))
                else "N/A"
            )

            col2.metric(
                "20-Day Support",
                f'{stock["support_20d"]:.2f}'
                if pd.notna(stock.get("support_20d"))
                else "N/A"
            )

            col3.metric(
                "20-Day Resistance",
                f'{stock["resistance_20d"]:.2f}'
                if pd.notna(stock.get("resistance_20d"))
                else "N/A"
            )

            # Technical levels chart
            chart_data = pd.DataFrame({
                "Level": [
                    "Support 2",
                    "Support 1",
                    "Pivot",
                    "Resistance 1",
                    "Resistance 2",
                    "20-Day Support",
                    "20-Day Resistance",
                ],
                "Price": [
                    stock.get("support_2"),
                    stock.get("support_1"),
                    stock.get("pivot"),
                    stock.get("resistance_1"),
                    stock.get("resistance_2"),
                    stock.get("support_20d"),
                    stock.get("resistance_20d"),
                ]
            })

            chart_data = chart_data.dropna(
                subset=["Price"]
            )

            fig = px.bar(
                chart_data,
                x="Level",
                y="Price",
                title=f"{selected_ticker} — Technical Price Levels",
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.write(
                "**Price Position:**",
                stock.get("price_position", "Unavailable")
            )

            st.write(
                "**Distance to Support:**",
                f'{stock["distance_to_support_pct"]:.2f}%'
                if pd.notna(stock.get("distance_to_support_pct"))
                else "N/A"
            )

            st.write(
                "**Distance to Resistance:**",
                f'{stock["distance_to_resistance_pct"]:.2f}%'
                if pd.notna(stock.get("distance_to_resistance_pct"))
                else "N/A"
            )

        else:
            st.warning(
                "Support and resistance data is unavailable. "
                "Run agents.calculate_support_resistance."
            )

        # Prediction chart
        if "Predicted Return (%)" in selected.columns:

            fig = px.bar(
                selected,
                x="ticker",
                y="Predicted Return (%)",
                color="predicted_direction",
                title=f"{selected_model} — Return Predictions",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.info(
            "Predictions and technical levels are analytical estimates, "
            "not investment advice. Support and resistance are based "
            "on historical prices and may change as new market data arrives. "
            "The sentiment model remains experimental due to limited "
            "social-data coverage."
        )



# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

elif page == "Sentiment Analysis":

    st.subheader("Multi-Source Sentiment Analysis")

    st.subheader("🏆 Social-only Stock Ranking (LIHKG + Telegram)")

    social_sentiment = all_sentiment[
        all_sentiment["source"].isin(["LIHKG", "Telegram"])
    ].copy() if not all_sentiment.empty else pd.DataFrame()

    if social_sentiment.empty:
        st.warning("No social sentiment records are available for ranking.")
    else:
        social_ranking = (
            social_sentiment.groupby("ticker")
            .agg(
                mentions=("dashboard_sentiment", "size"),
                positive=("dashboard_sentiment", lambda x: (x == "positive").sum()),
                negative=("dashboard_sentiment", lambda x: (x == "negative").sum()),
                neutral=("dashboard_sentiment", lambda x: (x == "neutral").sum()),
            )
        )

        # Reindex to every configured asset so missing social data is explicit.
        social_ranking = (
            social_ranking.reindex(ASSET_NAMES.keys(), fill_value=0)
            .rename_axis("ticker")
            .reset_index()
        )
        social_ranking["net_sentiment"] = (
            social_ranking["positive"] - social_ranking["negative"]
        )
        social_ranking["sentiment_score"] = np.where(
            social_ranking["mentions"] > 0,
            social_ranking["net_sentiment"] / social_ranking["mentions"],
            np.nan,
        )
        social_ranking["direction"] = np.select(
            [
                social_ranking["mentions"] == 0,
                social_ranking["net_sentiment"] > 0,
                social_ranking["net_sentiment"] < 0,
            ],
            ["NO DATA", "BULLISH", "BEARISH"],
            default="NEUTRAL",
        )
        social_ranking["company"] = social_ranking["ticker"].map(ASSET_NAMES)
        social_ranking["evidence_status"] = np.select(
            [
                social_ranking["mentions"] == 0,
                social_ranking["mentions"] >= 10,
            ],
            ["No social records", "10+ records in sample"],
            default="Fewer than 10 records",
        )
        social_ranking = social_ranking.sort_values(
            ["mentions", "sentiment_score"],
            ascending=[False, False],
        ).reset_index(drop=True)
        social_ranking.insert(0, "Rank", range(1, len(social_ranking) + 1))
        social_ranking = social_ranking.rename(columns={
            "ticker": "Ticker",
            "company": "Company",
            "mentions": "Social Mentions",
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "net_sentiment": "Net Sentiment",
            "sentiment_score": "Social Sentiment Score",
            "direction": "Sentiment Direction",
            "evidence_status": "Record Count",
        })
        st.dataframe(social_ranking, use_container_width=True, hide_index=True)
        fig = px.bar(
            social_ranking,
            x="Ticker",
            y="Social Sentiment Score",
            color="Sentiment Direction",
            title="Social-only Sentiment Score by HKEX Ticker",
            hover_data=["Company", "Social Mentions", "Positive", "Negative", "Neutral"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This ranking uses LIHKG and Telegram only. A missing ticker means no matching social posts were collected; it does not mean neutral sentiment."
        )

    st.divider()
    st.subheader("Combined Sentiment Ranking (Social + Exa News)")

    if all_sentiment.empty:
        st.warning("No usable sentiment records available.")
    else:
        ranking = (
            all_sentiment.groupby("ticker")
            .agg(
                mentions=("dashboard_sentiment", "size"),
                positive=("dashboard_sentiment", lambda x: (x == "positive").sum()),
                negative=("dashboard_sentiment", lambda x: (x == "negative").sum()),
                neutral=("dashboard_sentiment", lambda x: (x == "neutral").sum()),
            )
            .reset_index()
        )
        ranking["net_sentiment"] = ranking["positive"] - ranking["negative"]
        ranking["sentiment_score"] = ranking["net_sentiment"] / ranking["mentions"]
        ranking["direction"] = np.select(
            [ranking["net_sentiment"] > 0, ranking["net_sentiment"] < 0],
            ["BULLISH", "BEARISH"],
            default="NEUTRAL",
        )
        ranking["company"] = ranking["ticker"].map(ASSET_NAMES)
        ranking["evidence_status"] = np.where(
            ranking["mentions"] >= 10,
            "10+ records in sample",
            "Fewer than 10 records",
        )
        ranking = ranking.sort_values(
            ["sentiment_score", "mentions"], ascending=[False, False]
        ).reset_index(drop=True)
        ranking.insert(0, "Rank", range(1, len(ranking) + 1))
        ranking = ranking.rename(columns={
            "ticker": "Ticker",
            "company": "Company",
            "mentions": "Mentions",
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "net_sentiment": "Net Sentiment",
            "sentiment_score": "Sentiment Score",
            "direction": "Sentiment Direction",
            "evidence_status": "Record Count",
        })
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        fig = px.bar(
            ranking,
            x="Ticker",
            y="Sentiment Score",
            color="Sentiment Direction",
            title="Combined Social and News Sentiment Score",
            hover_data=["Company", "Mentions", "Positive", "Negative", "Neutral"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Combined score uses LIHKG, Telegram, and Exa headlines. Sentiment direction is derived from the net classified sentiment; it is not a price prediction or a trader position."
        )
    source_option = st.selectbox(
        "Select sentiment source",
        ["All Sources", "LIHKG", "Telegram", "Exa News"]
    )

    source_map = {
        "LIHKG": lihkg,
        "Telegram": telegram,
        "Exa News": exa,
    }

    if source_option == "All Sources":
        frames = [
            frame.assign(source=name)
            for name, frame in source_map.items()
            if not frame.empty
        ]

        sentiment_data = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )
    else:
        sentiment_data = source_map[source_option].copy()

    if sentiment_data.empty:
        st.warning(
            "No sentiment records available for this selection."
        )
    else:
        if "ticker" in sentiment_data.columns:
            available_tickers = sorted(
                sentiment_data["ticker"].dropna().unique()
            )

            selected_tickers = st.multiselect(
                "Filter stocks",
                available_tickers,
                default=available_tickers
            )

            sentiment_data = sentiment_data[
                sentiment_data["ticker"].isin(selected_tickers)
            ]

        counts = sentiment_counts(sentiment_data)

        col1, col2, col3 = st.columns(3)

        values = dict(
            zip(counts["Sentiment"], counts["Count"])
        )

        col1.metric("Positive", values.get("positive", 0))
        col2.metric("Negative", values.get("negative", 0))
        col3.metric("Neutral", values.get("neutral", 0))

        fig = px.pie(
            counts,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution",
            hole=0.45,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if "ticker" in sentiment_data.columns:
            grouped = (
                sentiment_data.groupby(
                    ["ticker", "dashboard_sentiment"]
                )
                .size()
                .reset_index(name="count")
            )

            fig = px.bar(
                grouped,
                x="ticker",
                y="count",
                color="dashboard_sentiment",
                barmode="group",
                title="Sentiment by Stock",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Sentiment Records")

        preferred_columns = [
            "ticker",
            "source",
            "dashboard_sentiment",
            "headline",
            "original_text",
            "created_at",
            "published_at",
            "url",
            "thread_url",
        ]

        visible = [
            c for c in preferred_columns
            if c in sentiment_data.columns
        ]

        st.dataframe(
            sentiment_data[visible],
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# HISTORICAL PRICES
# --------------------------------------------------

elif page == "Historical Prices":

    st.subheader("Historical Stock Prices")

    if unified.empty:
        st.warning("Historical price dataset unavailable.")
    else:
        prices = unified.copy()

        prices["Date"] = pd.to_datetime(
            prices["Date"],
            errors="coerce"
        )

        ticker_options = sorted(
            prices["ticker"].dropna().unique()
        )

        ticker = st.selectbox(
            "Select stock",
            ticker_options,
            format_func=lambda x: f"{x} — {ASSET_NAMES.get(x, x)}"
        )

        stock_data = prices[
            prices["ticker"] == ticker
        ].dropna(subset=["Date"]).sort_values("Date")

        if not stock_data.empty:
            fig = px.line(
                stock_data,
                x="Date",
                y="Close",
                title=f"{ASSET_NAMES.get(ticker, ticker)} — Historical Close",
                labels={"Close": "Adjusted Closing Price"},
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if "Volume" in stock_data.columns:
                volume_fig = px.bar(
                    stock_data,
                    x="Date",
                    y="Volume",
                    title="Trading Volume",
                )

                st.plotly_chart(
                    volume_fig,
                    use_container_width=True
                )

            st.subheader("Technical Indicators")

            indicator_columns = [
                "Date",
                "Close",
                "sma_5",
                "sma_10",
                "sma_20",
                "rsi_14",
            ]

            indicator_columns = [
                c for c in indicator_columns
                if c in stock_data.columns
            ]

            st.dataframe(
                stock_data[indicator_columns].tail(20),
                use_container_width=True,
                hide_index=True,
            )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------


elif page == "Model Evaluation":

    st.subheader("Model Performance Comparison")

    comparison_file = (
        MODELS_DIR / "fair_model_comparison.csv"
    )

    fair_results = load_csv(comparison_file)

    if fair_results.empty:

        st.warning(
            "Fair model comparison is not available. "
            "Run agents.evaluate_models_fairly first."
        )

    else:

        st.caption(
            "All models are evaluated on the same test observations."
        )

        st.dataframe(
            fair_results,
            use_container_width=True,
            hide_index=True
        )

        metric_options = {
            "MAE": "MAE",
            "RMSE": "RMSE",
            "Directional Accuracy": "Directional accuracy (%)",
        }

        selected_metric = st.selectbox(
            "Select evaluation metric",
            list(metric_options.keys())
        )

        metric_column = metric_options[selected_metric]

        fig = px.bar(
            fair_results,
            x="Model",
            y=metric_column,
            color="Model",
            title=f"Model Comparison — {selected_metric}",
            text_auto=".4f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Evaluation Summary")

        best_mae = fair_results.loc[
            fair_results["MAE"].idxmin()
        ]

        best_rmse = fair_results.loc[
            fair_results["RMSE"].idxmin()
        ]

        best_direction = fair_results.loc[
            fair_results[
                "Directional accuracy (%)"
            ].idxmax()
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Lowest MAE",
            best_mae["Model"]
        )

        col2.metric(
            "Lowest RMSE",
            best_rmse["Model"]
        )

        col3.metric(
            "Highest Directional Accuracy",
            best_direction["Model"],
            f'{best_direction["Directional accuracy (%)"]:.1f}%'
        )

        st.warning(
            "The evaluation set contains only five observations. "
            "These results are exploratory and cannot establish "
            "reliable predictive performance."
        )

        prediction_file = (
            MODELS_DIR / "fair_model_predictions.csv"
        )

        fair_predictions = load_csv(prediction_file)

        if not fair_predictions.empty:

            st.subheader("Actual vs Predicted Returns")

            fig = px.scatter(
                fair_predictions,
                x="target_next_return",
                y="predicted_next_return",
                color="model",
                hover_data=["Date", "ticker"],
                title="Actual vs Predicted Next-Session Returns",
                labels={
                    "target_next_return": "Actual Return",
                    "predicted_next_return": "Predicted Return",
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                fair_predictions,
                use_container_width=True,
                hide_index=True
            )


        st.warning(
            "The sentiment evaluation uses a very small sample. "
            "Compare models on identical test dates before drawing "
            "conclusions about whether sentiment improves predictions."
        )


# --------------------------------------------------
# DATA COVERAGE
# --------------------------------------------------

elif page == "Data Coverage":

    st.subheader("Data Collection Coverage")

    coverage = pd.DataFrame({
        "Source": [
            "LIHKG",
            "Telegram",
            "Exa News",
            "Historical Prices",
        ],
        "Available Records": [
            len(lihkg),
            len(telegram),
            len(exa),
            len(unified),
        ],
    })

    st.dataframe(
        coverage,
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        coverage,
        x="Source",
        y="Available Records",
        color="Source",
        title="Data Collection by Source",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Stock-wise Sentiment Coverage")

    if not latest_prices.empty:
        coverage_columns = [
            "ticker",
            "lihkg_count",
            "telegram_count",
            "exa_count",
            "total_sentiment_events",
        ]

        coverage_columns = [
            c for c in coverage_columns
            if c in latest_prices.columns
        ]

        st.dataframe(
            latest_prices[coverage_columns],
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "A low number of social posts does not indicate neutral sentiment. "
        "It indicates limited available social evidence."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "HengSeng Sentiment Price Prediction Agent | "
    "Research Prototype | HKEX Market Intelligence"
)
