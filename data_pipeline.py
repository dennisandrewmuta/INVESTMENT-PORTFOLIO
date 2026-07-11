"""
data_pipeline.py
----------------
Turns the raw Kaggle CSV (Ticker, Date, OHLCV) into a clean table where each row
is "one stock on one day" described by technical features, with a label saying
whether that stock went on to beat the market over the next HORIZON days.

The golden rule of this file:
    FEATURES use only PAST and PRESENT data.
    The LABEL uses FUTURE data (that's allowed — it's what we're trying to learn).
If a feature ever peeks into the future, your model is cheating and your whole
project is invalid. Every feature below is built from past prices only.
"""

import os
import numpy as np
import pandas as pd
import config


def load_data():
    """Load the CSV (or a fast cached copy), sorted by ticker then date."""
    if os.path.exists(config.CACHE_PATH):
        print(f"Loading cached data from {config.CACHE_PATH} ...")
        df = pd.read_pickle(config.CACHE_PATH)
    else:
        print(f"Reading {config.DATA_PATH} (this happens once) ...")
        # float32 instead of float64 halves the memory use — matters for 2.7M rows.
        df = pd.read_csv(
            config.DATA_PATH,
            dtype={
                "Open": "float32", "High": "float32", "Low": "float32",
                "Close": "float32", "Adj Close": "float32", "Volume": "int64",
            },
        )
        df["Date"] = pd.to_datetime(df["Date"])
        # Sorting is essential: rolling windows and shifts assume rows are in
        # time order within each ticker.
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df.to_pickle(config.CACHE_PATH)
        print(f"Cached cleaned data to {config.CACHE_PATH}")

    if config.SAMPLE_TICKERS:
        df = df[df["Ticker"].isin(config.SAMPLE_TICKERS)].copy()
        print(f"Using sample of {df['Ticker'].nunique()} tickers ({len(df):,} rows)")
    else:
        print(f"Loaded {df['Ticker'].nunique()} tickers, {len(df):,} rows")
    return df


def _rsi(series, n=14):
    """
    Relative Strength Index: a classic 0-100 momentum oscillator.
    Above ~70 = possibly overbought, below ~30 = possibly oversold.
    Built only from past price changes, so it's a safe feature.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def engineer_features(df):
    """Add all the predictor columns. Grouped by ticker so windows never
    bleed across two different companies."""
    print("Engineering features ...")
    g = df.groupby("Ticker", group_keys=False)

    # Past returns over different lookbacks (price change, no future info)
    df["ret_1"] = g["Adj Close"].pct_change(1)
    df["ret_5"] = g["Adj Close"].pct_change(5)
    df["ret_21"] = g["Adj Close"].pct_change(21)

    # Price relative to its own moving averages (>1 = trading above average)
    df["px_sma20"] = df["Adj Close"] / g["Adj Close"].transform(lambda s: s.rolling(20).mean())
    df["px_sma50"] = df["Adj Close"] / g["Adj Close"].transform(lambda s: s.rolling(50).mean())

    # Volatility: how jumpy the daily returns have been (a risk measure)
    df["vol_21"] = g["ret_1"].transform(lambda s: s.rolling(21).std())

    # Momentum: total move over the last ~3 months
    df["mom_63"] = g["Adj Close"].transform(lambda s: s / s.shift(63) - 1)

    # Volume relative to its recent norm (spikes can signal interest)
    df["volume_ratio"] = df["Volume"] / g["Volume"].transform(lambda s: s.rolling(20).mean())

    # RSI momentum oscillator
    df["rsi_14"] = g["Adj Close"].transform(lambda s: _rsi(s))

    return df


def build_labels(df):
    """
    Create the thing we're trying to predict.

    Step 1: forward return = how much the stock moves over the NEXT HORIZON days.
            This LEGITIMATELY uses future data — it's the answer key.
    Step 2: turn it into a cross-sectional label: 1 if this stock beat the median
            stock on that same date, else 0. This removes the overall market drift
            and leaves the pure "which stocks are relatively better" signal.
    """
    print("Building cross-sectional labels ...")
    g = df.groupby("Ticker", group_keys=False)
    h = config.HORIZON

    # shift(-h) looks h days into the FUTURE — only ever used for the label.
    df["fwd_ret"] = g["Adj Close"].transform(lambda s: s.shift(-h) / s - 1)

    # median forward return across all stocks on each date
    daily_median = df.groupby("Date")["fwd_ret"].transform("median")
    df["target"] = (df["fwd_ret"] > daily_median).astype("int8")
    return df


def prepare_dataset():
    """Full pipeline: load -> features -> labels -> drop rows with missing values."""
    df = load_data()
    df = engineer_features(df)
    df = build_labels(df)

    needed = config.FEATURES + ["target", "fwd_ret"]
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} rows with missing values "
          f"(warm-up periods and the final {config.HORIZON} days per stock).")
    print(f"Final dataset: {len(df):,} rows. "
          f"Label balance (fraction beating market): {df['target'].mean():.3f}")
    return df


if __name__ == "__main__":
    # Quick standalone check: build the dataset and show a peek.
    data = prepare_dataset()
    print("\nSample of engineered rows:")
    cols = ["Ticker", "Date"] + config.FEATURES + ["target"]
    print(data[cols].tail(5).to_string(index=False))
