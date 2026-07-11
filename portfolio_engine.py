"""
portfolio_engine.py
-------------------
The financial logic that sits between the ML model and the user.

The ML model (from train_model.py) only does ONE job: score every stock by how
likely it is to beat the market. This module turns those scores into an actual
investment decision the user asked for:

  * apply_risk_profile() - pick & weight stocks to match Low / Medium / High risk
  * allocate_money()      - split the user's money across the chosen stocks
  * portfolio_returns()   - reconstruct the portfolio's historical daily returns
  * risk_metrics()        - volatility, return, Sharpe ratio, worst drawdown
  * monte_carlo_forecast()- simulate where the investment could end up after N days

Why Monte Carlo for the forecast?
The model predicts over a fixed 21-day window. But the user wants to forecast an
arbitrary number of days. Instead of pretending the model can do that, we simulate
thousands of possible futures using the selected portfolio's OWN historical daily
behaviour. That is honest, flexible, and exactly how real risk teams forecast.
"""

import numpy as np
import pandas as pd

# How each risk appetite changes stock selection and weighting.
#   n             = how many stocks to hold (more = more diversified = safer)
#   candidate_pool= how many of the model's top picks to choose from
#   vol_tilt      = within that pool, prefer 'low', 'high', or 'none' volatility
#   weighting     = 'inverse_vol' (risk-balanced) or 'score' (conviction-weighted)
RISK_CONFIG = {
    "Low":    dict(n=25, candidate_pool=80, vol_tilt="low",  weighting="inverse_vol"),
    "Medium": dict(n=15, candidate_pool=40, vol_tilt="none", weighting="score"),
    "High":   dict(n=8,  candidate_pool=25, vol_tilt="high", weighting="score"),
}


def apply_risk_profile(scores, risk):
    """
    Choose and weight stocks to match the risk appetite.

    `scores` is the table of all stocks already scored by the model
    (must contain: Ticker, score, vol_21, Adj Close), sorted best-first.
    Returns the selected picks with a 'weight' column that sums to 1.
    """
    cfg = RISK_CONFIG[risk]

    # 1. Start from the model's strongest picks (its highest scores).
    pool = scores.head(cfg["candidate_pool"]).copy()

    # 2. Tilt toward calmer or punchier stocks depending on appetite.
    if cfg["vol_tilt"] == "low":
        picks = pool.nsmallest(cfg["n"], "vol_21").copy()      # steadiest names
    elif cfg["vol_tilt"] == "high":
        picks = pool.nlargest(cfg["n"], "vol_21").copy()       # most volatile names
    else:
        picks = pool.head(cfg["n"]).copy()                     # purely by score

    # 3. Decide how much weight each stock gets.
    if cfg["weighting"] == "inverse_vol":
        # Lower-volatility stocks get MORE weight -> a calmer portfolio.
        inv = 1.0 / picks["vol_21"].replace(0, np.nan)
        picks["weight"] = inv / inv.sum()
    else:
        # Weight by conviction: how far above 0.5 the model's score is.
        edge = (picks["score"] - 0.5).clip(lower=0)
        picks["weight"] = edge / edge.sum() if edge.sum() > 0 else 1.0 / len(picks)

    return picks.sort_values("weight", ascending=False).reset_index(drop=True)


def allocate_money(picks, amount):
    """Split `amount` (USD) across the picks by weight, and work out whole shares."""
    picks = picks.copy()
    picks["allocation"] = picks["weight"] * amount
    picks["shares"] = (picks["allocation"] / picks["Adj Close"]).apply(np.floor).astype(int)
    picks["invested"] = picks["shares"] * picks["Adj Close"]
    return picks


def portfolio_returns(picks, featured_df, lookback_days=504):
    """
    Rebuild the portfolio's daily return history using the chosen weights.

    We take the daily returns (ret_1) of each held stock over the last
    `lookback_days` (~2 trading years), weight them, and sum across the row.
    Because we use the stocks' ACTUAL joint history, correlations between them
    are captured automatically.
    """
    held = picks["Ticker"].tolist()
    weights = picks.set_index("Ticker")["weight"]

    sub = featured_df[featured_df["Ticker"].isin(held)][["Date", "Ticker", "ret_1"]]
    wide = sub.pivot(index="Date", columns="Ticker", values="ret_1").dropna()
    wide = wide.tail(lookback_days)

    # weighted sum of each stock's daily return = portfolio daily return
    aligned_w = weights.reindex(wide.columns).fillna(0)
    port_ret = (wide * aligned_w).sum(axis=1)
    return port_ret


def risk_metrics(port_ret):
    """Standard risk/return statistics for the portfolio's daily return series."""
    mean_d = port_ret.mean()
    std_d = port_ret.std()

    annual_return = mean_d * 252
    annual_vol = std_d * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    # worst peak-to-trough fall over the lookback window
    curve = (1 + port_ret).cumprod()
    drawdown = curve / curve.cummax() - 1
    max_drawdown = drawdown.min()

    return dict(
        daily_mean=mean_d,
        daily_vol=std_d,
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
    )


def classify_realized_risk(annual_vol):
    """Turn annualised volatility into a plain-language risk band."""
    if annual_vol < 0.18:
        return "Low", "The portfolio's swings are modest."
    elif annual_vol < 0.30:
        return "Medium", "The portfolio has moderate ups and downs."
    else:
        return "High", "The portfolio is volatile — big gains and big drops are likely."


def monte_carlo_forecast(port_ret, amount, days, n_sims=5000, seed=42):
    """
    Simulate `n_sims` possible futures for the investment over `days` days.

    Method: bootstrap. For each simulated future we randomly resample real
    historical daily returns (with replacement) and compound them. This keeps
    the real shape of the return distribution (including fat tails) instead of
    assuming a tidy bell curve.

    Returns:
      final_values : array of simulated end values (one per simulation)
      bands        : DataFrame of value over time at the 5/25/50/75/95 percentiles
    """
    rng = np.random.default_rng(seed)
    history = port_ret.values
    history = history[~np.isnan(history)]

    # (n_sims x days) matrix of resampled daily returns
    draws = rng.choice(history, size=(n_sims, days), replace=True)
    paths = amount * np.cumprod(1 + draws, axis=1)          # value each day
    paths = np.hstack([np.full((n_sims, 1), amount), paths])  # start at day 0

    final_values = paths[:, -1]

    pcts = [5, 25, 50, 75, 95]
    bands = pd.DataFrame(
        {f"p{p}": np.percentile(paths, p, axis=0) for p in pcts}
    )
    bands.index.name = "Day"
    return final_values, bands


def forecast_summary(final_values, amount):
    """Headline numbers from the simulation, ready to display."""
    expected = np.median(final_values)
    p5, p95 = np.percentile(final_values, [5, 95])
    prob_profit = float((final_values > amount).mean())
    return dict(
        expected_value=expected,
        expected_return=expected / amount - 1,
        worst_case=p5,            # 5th percentile ("bad but plausible")
        best_case=p95,            # 95th percentile ("good but plausible")
        prob_profit=prob_profit,
    )
