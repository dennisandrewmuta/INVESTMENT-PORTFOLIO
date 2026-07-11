"""
web_service.py
--------------
Bridge between the Flask server and the analytics code.

It runs in one of two modes, decided automatically at startup:

  * BUNDLE mode (production): loads the tiny precomputed bundle in website/data/
    built by build_artifacts.py. Uses a few MB of memory — fits a free host.
  * FULL mode (local fallback): if the bundle is missing, it loads the model and
    full dataset like the prototype does. Convenient for local development.

Saved portfolios are stored per browser session via web_db (SQLite or Postgres).
"""

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import config
import portfolio_engine as pe
import web_db

DATA = os.path.join(HERE, "data")
_STATE = {"scores": None, "returns": None, "backtest": None, "as_of": None, "mode": None}


def boot():
    if _STATE["mode"] is not None:
        return
    web_db.init_db()
    if os.path.exists(os.path.join(DATA, "scores.parquet")):
        _boot_bundle()
    else:
        _boot_full()


def _boot_bundle():
    _STATE["scores"] = pd.read_parquet(os.path.join(DATA, "scores.parquet"))
    _STATE["returns"] = pd.read_parquet(os.path.join(DATA, "returns.parquet"))
    with open(os.path.join(DATA, "backtest.json")) as f:
        _STATE["backtest"] = json.load(f)
    with open(os.path.join(DATA, "meta.json")) as f:
        _STATE["as_of"] = json.load(f)["as_of"]
    _STATE["mode"] = "bundle"
    print(f"[web_service] BUNDLE mode — {len(_STATE['scores'])} stocks, as of {_STATE['as_of']}")


def _boot_full():
    import joblib
    from data_pipeline import load_data, engineer_features
    os.chdir(ROOT)
    model = joblib.load(config.MODEL_OUT)
    featured = engineer_features(load_data())
    scored = featured.dropna(subset=config.FEATURES)
    idx = scored.groupby("Ticker")["Date"].idxmax()
    snap = scored.loc[idx].reset_index(drop=True)
    snap["score"] = model["model"].predict_proba(snap[config.FEATURES])[:, 1]
    snap["rank"] = snap["score"].rank(ascending=False).astype(int)
    snap = snap.sort_values("score", ascending=False).reset_index(drop=True)
    _STATE["scores"] = snap[["Ticker", "Adj Close", "vol_21", "score", "rank"]]
    _STATE["returns"] = featured[["Date", "Ticker", "ret_1"]].dropna()
    _STATE["as_of"] = str(snap["Date"].max().date())
    _STATE["_model"] = model
    _STATE["_featured"] = featured
    _STATE["mode"] = "full"
    print("[web_service] FULL mode (local fallback)")


def _f(x):
    return float(x)


def build_portfolio(amount, risk, horizon):
    scores = _STATE["scores"]
    returns = _STATE["returns"]

    picks = pe.apply_risk_profile(scores, risk)
    picks = pe.allocate_money(picks, amount)
    port_ret = pe.portfolio_returns(picks, returns)
    metrics = pe.risk_metrics(port_ret)
    final_values, bands = pe.monte_carlo_forecast(port_ret, amount, horizon)
    fsum = pe.forecast_summary(final_values, amount)
    band_label, band_note = pe.classify_realized_risk(metrics["annual_vol"])

    holdings = [{
        "ticker": r["Ticker"], "weight": _f(r["weight"]), "allocation": _f(r["allocation"]),
        "shares": int(r["shares"]), "price": _f(r["Adj Close"]), "score": _f(r["score"]),
    } for _, r in picks.iterrows()]

    counts, edges = np.histogram(final_values, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2

    return {
        "as_of": _STATE["as_of"],
        "inputs": {"amount": amount, "risk": risk, "horizon": horizon},
        "invested": _f(picks["invested"].sum()),
        "cash_left": _f(amount - picks["invested"].sum()),
        "n_stocks": len(picks),
        "holdings": holdings,
        "risk": {
            "annual_vol": _f(metrics["annual_vol"]), "annual_return": _f(metrics["annual_return"]),
            "sharpe": _f(metrics["sharpe"]), "max_drawdown": _f(metrics["max_drawdown"]),
            "band": band_label, "band_note": band_note,
        },
        "value_curve": [_f(v) for v in ((1 + port_ret).cumprod() * amount).tail(504).values],
        "forecast": {
            "expected_value": _f(fsum["expected_value"]), "expected_return": _f(fsum["expected_return"]),
            "worst_case": _f(fsum["worst_case"]), "best_case": _f(fsum["best_case"]),
            "prob_profit": _f(fsum["prob_profit"]),
            "days": list(range(len(bands))),
            "p5": [_f(v) for v in bands["p5"].values],
            "p50": [_f(v) for v in bands["p50"].values],
            "p95": [_f(v) for v in bands["p95"].values],
            "hist_x": [_f(v) for v in centers], "hist_y": [int(v) for v in counts],
        },
    }


def get_backtest():
    if _STATE["backtest"] is not None:
        return _STATE["backtest"]
    # FULL mode: compute once
    from data_pipeline import build_labels
    model = _STATE["_model"]
    df = build_labels(_STATE["_featured"].copy()).dropna(subset=config.FEATURES + ["target", "fwd_ret"])
    test = df[df["Date"] >= pd.Timestamp(config.TRAIN_END_DATE)].copy()
    test["score"] = model["model"].predict_proba(test[config.FEATURES])[:, 1]
    test["decile"] = pd.qcut(test["score"], 10, labels=False, duplicates="drop")
    dec = test.groupby("decile")["fwd_ret"].mean()
    _STATE["backtest"] = {"deciles": [round(_f(v) * 100, 2) for v in dec.values],
                          "market": round(_f(test["fwd_ret"].mean()) * 100, 2),
                          "top": round(_f(dec.iloc[-1]) * 100, 2), "bottom": round(_f(dec.iloc[0]) * 100, 2)}
    return _STATE["backtest"]


def save_portfolio(session_id, data):
    inp, r, f = data["inputs"], data["risk"], data["forecast"]
    m = dict(label=f"{inp['risk']} · ${inp['amount']:,.0f} · {inp['horizon']}d",
             amount=inp["amount"], risk=inp["risk"], horizon=inp["horizon"],
             n_stocks=data["n_stocks"], invested=data["invested"],
             annual_vol=r["annual_vol"], annual_return=r["annual_return"], sharpe=r["sharpe"],
             max_drawdown=r["max_drawdown"], expected_value=f["expected_value"],
             expected_return=f["expected_return"], worst_case=f["worst_case"],
             best_case=f["best_case"], prob_profit=f["prob_profit"])
    return web_db.save_portfolio(session_id, m, data["holdings"])


def list_saved(session_id):
    rows = web_db.list_portfolios(session_id)
    for r in rows:
        if r.get("created_at") is not None:
            r["created_at"] = str(r["created_at"])
    return rows


def clear_saved(session_id):
    web_db.clear_session(session_id)


def as_of():
    return _STATE["as_of"]
