"""
predict.py
----------
Uses the trained model to build an actual investment portfolio from the most
recent data. This is the 'system' part of your Smart Investment Portfolio System:
training produced a model; this turns the model into stock picks.

Run it with:   python predict.py

What it does:
  1. Load the saved model
  2. Rebuild features on the full dataset
  3. Take the LATEST available day for every stock
  4. Score each stock with the model (probability it beats the market)
  5. Build a portfolio: pick the top-N stocks and weight them
"""

import numpy as np
import pandas as pd
import joblib

import config
from data_pipeline import load_data, engineer_features

TOP_N = 20            # how many stocks to hold
WEIGHTING = "score"   # "equal" = same weight each, "score" = weight by confidence


def latest_rows(df):
    """Keep only the most recent row per ticker that has all features present."""
    df = df.dropna(subset=config.FEATURES)
    # the last (most recent) row for each ticker
    idx = df.groupby("Ticker")["Date"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_portfolio():
    saved = joblib.load(config.MODEL_OUT)
    model, features = saved["model"], saved["features"]

    df = load_data()
    df = engineer_features(df)
    snapshot = latest_rows(df)

    print(f"Scoring {len(snapshot)} stocks as of their latest available date "
          f"({snapshot['Date'].max().date()}) ...")
    snapshot["score"] = model.predict_proba(snapshot[features])[:, 1]

    ranked = snapshot.sort_values("score", ascending=False)
    picks = ranked.head(TOP_N).copy()

    if WEIGHTING == "equal":
        picks["weight"] = 1.0 / len(picks)
    else:  # weight proportional to how far above 0.5 the score is
        edge = (picks["score"] - 0.5).clip(lower=0)
        picks["weight"] = edge / edge.sum() if edge.sum() > 0 else 1.0 / len(picks)

    print(f"\n=== RECOMMENDED PORTFOLIO (top {TOP_N} stocks) ===")
    print(f"{'Ticker':<8}{'Score':>8}{'Weight':>10}{'Last Price':>12}")
    for _, r in picks.iterrows():
        print(f"{r['Ticker']:<8}{r['score']:>8.3f}{r['weight']:>9.1%}{r['Adj Close']:>12.2f}")

    print("\nReminder: this is an academic model, not financial advice. The edge is")
    print("statistical and only shows up across the whole basket over many periods.")

    picks[["Ticker", "Date", "Adj Close", "score", "weight"]].to_csv(
        "recommended_portfolio.csv", index=False)
    print("Saved recommended_portfolio.csv")
    return picks


if __name__ == "__main__":
    build_portfolio()
