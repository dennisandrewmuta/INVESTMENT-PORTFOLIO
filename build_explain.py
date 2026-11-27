"""
build_explain.py
----------------
Builds website/data/explain.json: for every stock, WHY the model ranked it where
it did. The website reads this file to draw the "Why these picks?" panel.

Two methods, chosen automatically (the file records which one was used):

  method = "model"      EXACT. Uses the trained model. For each stock and each
                        feature we replace that one feature by the typical
                        (median) stock's value and see how much the model's score
                        changes. Run through build_artifacts.py, which has the model.

  method = "surrogate"  APPROXIMATE. Used when only the small saved data bundle is
                        available (no model, no raw CSV). A transparent stand-in
                        model is fitted to reproduce the real model's saved scores
                        from features rebuilt out of the saved daily returns, then
                        explained the same way. Its fit (R-squared) is recorded so
                        the site can be honest about how closely it tracks the model.
                        The volume feature is not in the bundle, so it is left out.

Usage (bundle only; needs scikit-learn):   python build_explain.py
"""

import os
import re
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "website", "data")

# label, one-line meaning (shown on the site)
FEATURE_INFO = {
    "ret_1":        ("Yesterday's move",        "Return over the last trading day"),
    "ret_5":        ("1-week return",           "Return over the last 5 trading days"),
    "ret_21":       ("1-month return",          "Return over the last 21 trading days"),
    "mom_63":       ("3-month momentum",        "Return over the last ~63 trading days"),
    "px_sma20":     ("Price vs 20-day average", "Above 1.0 = trading above its 20-day average"),
    "px_sma50":     ("Price vs 50-day average", "Above 1.0 = trading above its 50-day average"),
    "vol_21":       ("Volatility (21-day)",     "How jumpy daily returns have been"),
    "rsi_14":       ("RSI (14-day)",            "0-100 momentum gauge; above 70 = overbought, below 30 = oversold"),
    "volume_ratio": ("Volume vs normal",        "Today's volume relative to its 20-day average"),
}


def reconstruct_features(returns_long):
    """Rebuild the model's price-based features from saved daily returns.
    Everything here is scale-free (ratios/percentages), so it matches the original
    features exactly; volume_ratio needs raw volume, which the bundle does not hold."""
    w = returns_long.pivot(index="Date", columns="Ticker", values="ret_1").sort_index().fillna(0.0)
    rel = (1 + w).cumprod()                                   # price path up to a constant

    def tail_ret(n):
        return (1 + w.tail(n)).prod() - 1

    F = pd.DataFrame(index=w.columns)
    F["ret_1"] = w.iloc[-1]
    F["ret_5"] = tail_ret(5)
    F["ret_21"] = tail_ret(21)
    F["mom_63"] = tail_ret(63)
    F["px_sma20"] = rel.iloc[-1] / rel.tail(20).mean()
    F["px_sma50"] = rel.iloc[-1] / rel.tail(50).mean()
    F["vol_21"] = w.tail(21).std()
    d = rel.diff().tail(14)
    gain, loss = d.clip(lower=0).mean(), (-d.clip(upper=0)).mean()
    F["rsi_14"] = 100 - 100 / (1 + gain / loss)
    return F.replace([np.inf, -np.inf], np.nan)


def _ablation(predict, X, median):
    """contribution[i, j] = score(x_i) - score(x_i with feature j set to the typical value)."""
    base = predict(X)
    out = np.zeros(X.shape, dtype=float)
    for j, col in enumerate(X.columns):
        Xj = X.copy()
        Xj[col] = median[col]
        out[:, j] = base - predict(Xj)
    return base, out


def surrogate_explain(F, scores):
    """Approximate explanation from the bundle alone."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    F = F.reindex(scores.index)
    X = F.fillna(F.median())
    y = scores["score"].values
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
    # shuffle: the scores table is sorted best-first, so unshuffled folds would be unrepresentative
    cv = cross_val_predict(gb, X.values, y, cv=KFold(5, shuffle=True, random_state=42))
    r2 = 1 - ((y - cv) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    gb.fit(X.values, y)
    predict = lambda D: gb.predict(D.values)
    base, contrib = _ablation(predict, X, X.median())
    return X, contrib, float(r2), float(np.median(base))


def model_explain(model, snap, features):
    """Exact explanation from the trained model. `snap` has one row per stock."""
    X = snap[features].astype(float)
    X = X.fillna(X.median())
    predict = lambda D: model.predict_proba(D)[:, 1]
    base, contrib = _ablation(predict, X, X.median())
    return X, contrib, None, float(np.median(base))


def _global_importance():
    """The real model's permutation importances, as saved by train_model.py in metrics.txt."""
    path = os.path.join(HERE, "metrics.txt")
    out = {}
    if os.path.exists(path):
        txt = open(path, encoding="utf-8").read()
        part = txt.split("Feature importance:")[-1]
        for k, v in re.findall(r"^\s*([a-z_0-9]+)\s+(-?\d+\.\d+)\s*$", part, flags=re.M):
            out[k] = float(v)
    return out


def write_explain(X, contrib, scores, method, r2, typical_score, as_of, out_dir=DATA):
    pctl = X.rank(pct=True).mul(100)
    stocks = {}
    for i, t in enumerate(X.index):
        stocks[t] = {
            "score": round(float(scores.loc[t, "score"]), 4),
            "rank": int(scores.loc[t, "rank"]),
            "val": {c: round(float(X.iloc[i][c]), 5) for c in X.columns},
            "pct": {c: round(float(pctl.iloc[i][c]), 1) for c in X.columns},
            "push": {c: round(float(contrib[i, j]), 5) for j, c in enumerate(X.columns)},
        }
    doc = {
        "method": method, "r2": None if r2 is None else round(r2, 3), "as_of": as_of,
        "n_stocks": len(stocks), "typical_score": round(typical_score, 4),
        "features": [{"key": k, "label": FEATURE_INFO[k][0], "desc": FEATURE_INFO[k][1]}
                     for k in X.columns],
        "global_importance": _global_importance(),
        "stocks": stocks,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "explain.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    return path, doc


def main():
    scores = pd.read_parquet(os.path.join(DATA, "scores.parquet")).set_index("Ticker")
    returns = pd.read_parquet(os.path.join(DATA, "returns.parquet"))
    with open(os.path.join(DATA, "meta.json")) as f:
        as_of = json.load(f)["as_of"]
    F = reconstruct_features(returns)
    X, contrib, r2, typical = surrogate_explain(F, scores)
    path, doc = write_explain(X, contrib, scores, "surrogate", r2, typical, as_of)
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)  method=surrogate  R2={r2:.3f}")


if __name__ == "__main__":
    main()
