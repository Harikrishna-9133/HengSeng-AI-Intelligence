
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/hkex_historical_prices.csv")
OUTPUT_FILE = Path(
    "data/processed/support_resistance_levels.csv"
)

WINDOW = 20


def calculate_levels(group):

    group = group.sort_values("Date").copy()

    if len(group) < WINDOW + 1:
        return None

    latest = group.iloc[-1]
    previous = group.iloc[-2]

    recent = group.tail(WINDOW)

    current_price = float(latest["Close"])

    # Rolling support and resistance.
    support_20d = float(recent["Low"].min())
    resistance_20d = float(recent["High"].max())

    # Standard pivot levels from the previous session.
    pivot = (
        float(previous["High"])
        + float(previous["Low"])
        + float(previous["Close"])
    ) / 3

    support_1 = (2 * pivot) - float(previous["High"])
    resistance_1 = (2 * pivot) - float(previous["Low"])

    support_2 = pivot - (
        float(previous["High"]) - float(previous["Low"])
    )

    resistance_2 = pivot + (
        float(previous["High"]) - float(previous["Low"])
    )

    return {
        "ticker": latest["ticker"],
        "company": latest["company"],
        "date": latest["Date"],
        "current_price": current_price,
        "support_20d": support_20d,
        "resistance_20d": resistance_20d,
        "pivot": pivot,
        "support_1": support_1,
        "resistance_1": resistance_1,
        "support_2": support_2,
        "resistance_2": resistance_2,
        "distance_to_support_pct": (
            (current_price - support_20d)
            / current_price
        ) * 100,
        "distance_to_resistance_pct": (
            (resistance_20d - current_price)
            / current_price
        ) * 100,
        "price_position": (
            "Near Support"
            if current_price <= support_20d * 1.02
            else "Near Resistance"
            if current_price >= resistance_20d * 0.98
            else "Between Levels"
        ),
    }


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Price dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "ticker",
        "company",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    for column in ["Open", "High", "Low", "Close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Date", "High", "Low", "Close", "ticker"]
    )

    results = []

    for ticker, group in df.groupby("ticker"):

        levels = calculate_levels(group)

        if levels:
            results.append(levels)

    if not results:
        raise ValueError(
            "No support/resistance levels could be calculated."
        )

    output = pd.DataFrame(results)

    numeric_columns = output.select_dtypes(
        include="number"
    ).columns

    output[numeric_columns] = output[
        numeric_columns
    ].round(4)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSUPPORT AND RESISTANCE ANALYSIS")
    print("=" * 65)

    print(
        output[
            [
                "ticker",
                "current_price",
                "support_20d",
                "resistance_20d",
                "pivot",
                "support_1",
                "resistance_1",
                "price_position",
            ]
        ].to_string(index=False)
    )

    print("\nStocks analyzed:", len(output))
    print("Latest data date:", output["date"].max())
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
