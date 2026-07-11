"""
build_artifacts.py
------------------
Run this ONCE locally before deploying. It does the heavy work (loading the full
dataset, scoring every stock, running the backtest) and saves a tiny bundle that
the deployed website loads instead of the 2.7M-row dataset.

    python build_artifacts.py

Output (in website/data/):
    scores.parquet   - 472 rows: latest score + volatility + price per stock
    returns.parquet  - ~2 years of daily returns per stock (for risk + Monte Carlo)
    backtest.json    - precomputed decile returns (proof the model works)
    meta.json        - the 'as of' date

Why: the full pipeline needs ~1 GB of memory; this bundle is a few MB, so the
site runs comfortably on a free hosting tier. The model and raw CSV never need to
be deployed — only these artifacts.
"""

import os
import json
import joblib
import pandas as pd

import config
from data_pipeline import load_data, engineer_features, build_labels

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "website", "data")
LOOKBACK_DAYS = 540  # ~2 trading years, enough for risk + forecast


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading model and data (one-off) ...")
    model = joblib.load(config.MODEL_OUT)
    featured = engineer_features(load_data())

    # ---- scores snapshot: latest scored row per stock ----
    scored = featured.dropna(subset=config.FEATURES)
    idx = scored.groupby("Ticker")["Date"].idxmax()
    snap = scored.loc[idx].reset_index(drop=True)
    snap["score"] = model["model"].predict_proba(snap[config.FEATURES])[:, 1]
    snap["rank"] = snap["score"].rank(ascending=False).astype(int)
    snap = snap.sort_values("score", ascending=False).reset_index(drop=True)
    scores = snap[["Ticker", "Adj Close", "vol_21", "score", "rank"]].copy()
    scores.to_parquet(os.path.join(OUT, "scores.parquet"))
    as_of = str(snap["Date"].max().date())
    print(f"  scores.parquet: {len(scores)} stocks, as of {as_of}")

    # ---- returns history: last ~2 years of daily returns per stock ----
    cutoff = featured["Date"].max() - pd.Timedelta(days=int(LOOKBACK_DAYS * 1.5))
    ret = featured[featured["Date"] >= cutoff][["Date", "Ticker", "ret_1"]].dropna()
    ret.to_parquet(os.path.join(OUT, "returns.parquet"))
    print(f"  returns.parquet: {len(ret):,} rows")

    # ---- backtest: decile returns (out-of-sample) ----
    df = build_labels(featured.copy()).dropna(subset=config.FEATURES + ["target", "fwd_ret"])
    test = df[df["Date"] >= pd.Timestamp(config.TRAIN_END_DATE)].copy()
    test["score"] = model["model"].predict_proba(test[config.FEATURES])[:, 1]
    test["decile"] = pd.qcut(test["score"], 10, labels=False, duplicates="drop")
    dec = test.groupby("decile")["fwd_ret"].mean()
    backtest = {
        "deciles": [round(float(v) * 100, 2) for v in dec.values],
        "market": round(float(test["fwd_ret"].mean()) * 100, 2),
        "top": round(float(dec.iloc[-1]) * 100, 2),
        "bottom": round(float(dec.iloc[0]) * 100, 2),
    }
    with open(os.path.join(OUT, "backtest.json"), "w") as f:
        json.dump(backtest, f)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"as_of": as_of}, f)
    print(f"  backtest.json: edge +{round(backtest['top'] - backtest['bottom'], 2)}%")

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"Done. Bundle size: {total/1e6:.1f} MB  (in {OUT})")


if __name__ == "__main__":
    main()
