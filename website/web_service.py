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
import market_data

DATA = os.path.join(HERE, "data")
_STATE = {"scores": None, "returns": None, "backtest": None, "as_of": None, "mode": None, "explain": None}


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
    ex = os.path.join(DATA, "explain.json")           # optional: "why was it ranked" data
    if os.path.exists(ex):
        with open(ex, encoding="utf-8") as f:
            _STATE["explain"] = json.load(f)
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


def _scores_with_live_prices():
    """Saved scores, with each stock's price replaced by its latest live price.
    Scores/volatility stay as the model computed them; only prices move, so share
    counts and allocations reflect today's market. Falls back per-stock to the
    saved price if a live price is missing or looks wrong (e.g. a stock split)."""
    base = _STATE["scores"]
    live, info = market_data.live_prices(base["Ticker"].tolist())
    if not live:
        return base, {"active": False, "prices_as_of": None, "live_count": 0, "total": len(base)}
    df = base.copy()
    df["Adj Close"] = df["Adj Close"].astype("float64")   # bundle stores float32
    df["model_price"] = df["Adj Close"]
    new = df["Ticker"].map(live).astype("float64")
    ok = new.notna() & (new / df["Adj Close"]).between(0.35, 3.0)
    df["Adj Close"] = np.where(ok, new, df["Adj Close"])
    return df, {"active": bool(ok.any()), "prices_as_of": info["date"],
                "live_count": int(ok.sum()), "total": len(df)}


def build_portfolio(amount, risk, horizon):
    scores, live = _scores_with_live_prices()
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
        "model_price": _f(r["model_price"]) if "model_price" in picks.columns else _f(r["Adj Close"]),
    } for _, r in picks.iterrows()]

    counts, edges = np.histogram(final_values, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2

    return {
        "as_of": _STATE["as_of"],
        "live": live,
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


# ----------------------------------------------------------------------------
# Interactive builder: one "profile" per risk level holds everything the page
# needs to recompute a portfolio for ANY amount and horizon in the browser.
#   * weights/scores/risk metrics do not depend on amount or horizon
#   * simulated values scale linearly with the amount invested
#   * one simulation to HZ_MAX days gives every shorter horizon too
# So the heavy work (the Monte Carlo) runs once per risk level, then is cached.
# ----------------------------------------------------------------------------
HZ_MAX, HZ_STEP, N_SIMS = 250, 5, 5000
_PROFILE_CACHE = {}


def _unit_profile(risk):
    """Heavy, price-independent part. Cached for the life of the process."""
    if risk in _PROFILE_CACHE:
        return _PROFILE_CACHE[risk]
    if risk == "Market":
        # benchmark: the whole S&P 500 universe, equal-weighted (same data, same method)
        wide = _STATE["returns"].pivot(index="Date", columns="Ticker", values="ret_1").sort_index()
        port_ret = wide.mean(axis=1).dropna().tail(504)
        tickers = []
    else:
        picks = pe.apply_risk_profile(_STATE["scores"], risk)
        port_ret = pe.portfolio_returns(picks, _STATE["returns"])
        tickers = picks[["Ticker", "weight", "score"]].to_dict("records")
    m = pe.risk_metrics(port_ret)
    band, note = pe.classify_realized_risk(m["annual_vol"])

    rng = np.random.default_rng(42)
    hist = port_ret.values
    hist = hist[~np.isnan(hist)]
    draws = rng.choice(hist, size=(N_SIMS, HZ_MAX), replace=True)
    paths = np.hstack([np.ones((N_SIMS, 1)), np.cumprod(1 + draws, axis=1)])   # value of $1, day 0..HZ_MAX

    pct = {f"p{p}": np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)}
    prob = (paths > 1).mean(axis=0)
    hz = list(range(HZ_STEP, HZ_MAX + 1, HZ_STEP))
    grid = np.arange(0, 101)
    quant = [np.percentile(paths[:, h], grid) for h in hz]      # final-value distribution per horizon

    def r(a, n):
        return [round(float(v), n) for v in a]

    prof = {
        "tickers": tickers,
        "metrics": {"annual_vol": _f(m["annual_vol"]), "annual_return": _f(m["annual_return"]),
                    "sharpe": _f(m["sharpe"]), "max_drawdown": _f(m["max_drawdown"]),
                    "band": band, "band_note": note},
        "curve": r((1 + port_ret).cumprod().tail(504).values, 5),
        "bands": {k: r(v, 5) for k, v in pct.items()},
        "prob": r(prob, 4),
        "hz": hz,
        "q": [r(row, 5) for row in quant],
        "n_sims": N_SIMS,
    }
    _PROFILE_CACHE[risk] = prof
    return prof


def build_profile(risk):
    """Profile for the browser: cached simulation + CURRENT prices for the holdings."""
    prof = _unit_profile(risk)
    scores, live = _scores_with_live_prices()
    price = dict(zip(scores["Ticker"], scores["Adj Close"]))
    model = dict(zip(scores["Ticker"], scores["model_price"])) if "model_price" in scores.columns else price
    ex = _STATE.get("explain")
    holdings = []
    for t in prof["tickers"]:
        h = {"ticker": t["Ticker"], "weight": float(t["weight"]), "score": float(t["score"]),
             "price": float(price[t["Ticker"]]), "model_price": float(model[t["Ticker"]])}
        if ex and t["Ticker"] in ex["stocks"]:
            w = ex["stocks"][t["Ticker"]]
            h["why"] = {"rank": w["rank"], "val": w["val"], "pct": w["pct"], "push": w["push"]}
        holdings.append(h)
    meta = None
    if ex:
        meta = {"method": ex["method"], "r2": ex["r2"], "as_of": ex["as_of"], "features": ex["features"],
                "typical_score": ex["typical_score"], "n_stocks": ex["n_stocks"],
                "global_importance": ex["global_importance"]}
    return {"risk": risk, "as_of": _STATE["as_of"], "live": live,
            "benchmark": risk == "Market", "explain": meta,
            "n_stocks": len(_STATE["scores"]) if risk == "Market" else len(holdings),
            "holdings": holdings, "metrics": prof["metrics"], "curve": prof["curve"],
            "bands": prof["bands"], "prob": prof["prob"], "hz": prof["hz"], "q": prof["q"],
            "n_sims": prof["n_sims"], "hz_max": HZ_MAX}


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


def market_snapshot():
    """Index levels + feed status for the ticker strip (never blocks)."""
    return market_data.snapshot(_STATE["scores"]["Ticker"].tolist())
