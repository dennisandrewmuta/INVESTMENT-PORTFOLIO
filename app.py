"""
app.py  -  Smart Investment Portfolio System (Streamlit prototype)
------------------------------------------------------------------
Live dashboard for the panel demo. The user enters how much to invest, picks a
risk appetite, and chooses a forecast horizon. The app then shows four sections:

    Recommendations -> Risk Analysis -> Prediction -> Conclusion

plus a stock lookup and the model backtest.

How the pieces fit together:
  * train_model.py        trained the model (stock SELECTION only)
  * portfolio_engine.py   turns scores into a risk-matched, money-allocated plan
                          and forecasts the investment with Monte Carlo simulation

Run it with:   streamlit run app.py
(Run `python train_model.py` once first to create the model file.)
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

import config
from data_pipeline import load_data, engineer_features, build_labels
import portfolio_engine as pe
import database as db

db.init_db()  # create the SQLite tables on first run

st.set_page_config(page_title="Smart Investment Portfolio System",
                   page_icon="📈", layout="wide")
st.title("📈 Smart Investment Portfolio System")
st.caption("Big Data Analytics on 26 years of S&P 500 data · final-year prototype")


# ----------------------------------------------------------------------
# Cached loaders (run once, then instant)
# ----------------------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(config.MODEL_OUT):
        return None
    return joblib.load(config.MODEL_OUT)


@st.cache_data(show_spinner="Loading data and building features ...")
def get_featured_data():
    df = load_data()
    return engineer_features(df)


@st.cache_data(show_spinner="Scoring stocks ...")
def get_latest_scores():
    model = get_model()
    df = get_featured_data().dropna(subset=config.FEATURES)
    idx = df.groupby("Ticker")["Date"].idxmax()
    snap = df.loc[idx].reset_index(drop=True)
    snap["score"] = model["model"].predict_proba(snap[config.FEATURES])[:, 1]
    snap["rank"] = snap["score"].rank(ascending=False).astype(int)
    return snap.sort_values("score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Running the backtest ...")
def get_backtest():
    model = get_model()
    df = build_labels(get_featured_data().copy())
    df = df.dropna(subset=config.FEATURES + ["target", "fwd_ret"])
    test = df[df["Date"] >= pd.Timestamp(config.TRAIN_END_DATE)].copy()
    test["score"] = model["model"].predict_proba(test[config.FEATURES])[:, 1]
    test["decile"] = pd.qcut(test["score"], 10, labels=False, duplicates="drop")
    return test.groupby("decile")["fwd_ret"].mean(), test["fwd_ret"].mean()


# ----------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------
model = get_model()
if model is None:
    st.error("No trained model found. Run `python train_model.py` first to create "
             f"`{config.MODEL_OUT}`, then reload this page.")
    st.stop()

scores = get_latest_scores()
featured = get_featured_data()
as_of = scores["Date"].max().date()

# ----------------------------------------------------------------------
# Sidebar: the three user inputs
# ----------------------------------------------------------------------
st.sidebar.header("Your investment")
amount = st.sidebar.number_input("Amount to invest (USD)", min_value=100,
                                 max_value=10_000_000, value=10_000, step=500)
risk = st.sidebar.radio("Risk appetite", ["Low", "Medium", "High"], index=1,
                        help="Low = many steady stocks, risk-balanced. "
                             "High = few volatile stocks, conviction-weighted.")
horizon = st.sidebar.slider("Forecast horizon (trading days)", 5, 252, 60, step=5,
                            help="~21 days = 1 month, ~252 = 1 year")
st.sidebar.markdown("---")
st.sidebar.caption(f"Prices as of {as_of}. S&P 500 stocks are priced in USD. "
                   "Academic model — not financial advice.")

# ----------------------------------------------------------------------
# Run the engine with the chosen inputs
# ----------------------------------------------------------------------
picks = pe.apply_risk_profile(scores, risk)
picks = pe.allocate_money(picks, amount)
port_ret = pe.portfolio_returns(picks, featured)
metrics = pe.risk_metrics(port_ret)
final_values, bands = pe.monte_carlo_forecast(port_ret, amount, horizon)
fsum = pe.forecast_summary(final_values, amount)
realized_band, realized_note = pe.classify_realized_risk(metrics["annual_vol"])

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
t_rec, t_risk, t_pred, t_conc, t_comp, t_look, t_back = st.tabs(
    ["💰 Recommendations", "⚠️ Risk Analysis", "🔮 Prediction",
     "📝 Conclusion", "🗄️ Compare Saved", "🔎 Stock Lookup", "📊 Model Backtest"])

# ---- Recommendations --------------------------------------------------
with t_rec:
    st.subheader(f"Recommended {risk}-risk portfolio · as of {as_of}")
    invested = picks["invested"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("You invest", f"${amount:,.0f}")
    c2.metric("Stocks held", len(picks))
    c3.metric("Actually deployed", f"${invested:,.0f}",
              help="Whole shares only, so a little cash is left over")
    c4.metric("Cash left over", f"${amount - invested:,.0f}")

    left, right = st.columns([3, 2])
    with left:
        tbl = picks[["Ticker", "weight", "allocation", "shares", "Adj Close", "score"]].copy()
        tbl.columns = ["Ticker", "Weight", "Allocation", "Shares", "Price", "Score"]
        tbl["Weight"] = tbl["Weight"].map("{:.1%}".format)
        tbl["Allocation"] = tbl["Allocation"].map("${:,.0f}".format)
        tbl["Price"] = tbl["Price"].map("${:.2f}".format)
        tbl["Score"] = tbl["Score"].map("{:.3f}".format)
        st.dataframe(tbl, hide_index=True, width='stretch')
    with right:
        st.caption("Money allocated per stock")
        st.bar_chart(picks.set_index("Ticker")["allocation"])

    st.download_button("⬇ Download plan as CSV",
                       picks[["Ticker", "weight", "allocation", "shares",
                              "Adj Close", "score"]].to_csv(index=False),
                       file_name=f"portfolio_{risk.lower()}_{amount}.csv",
                       mime="text/csv")

    st.markdown("---")
    save_col1, save_col2 = st.columns([2, 3])
    with save_col1:
        if st.button("💾 Save this portfolio", type="primary"):
            meta = dict(
                label=f"{risk} · ${amount:,.0f} · {horizon}d",
                amount=amount, risk=risk, horizon=horizon, n_stocks=len(picks),
                invested=float(invested), annual_vol=metrics["annual_vol"],
                annual_return=metrics["annual_return"], sharpe=metrics["sharpe"],
                max_drawdown=metrics["max_drawdown"],
                expected_value=fsum["expected_value"],
                expected_return=fsum["expected_return"],
                worst_case=fsum["worst_case"], best_case=fsum["best_case"],
                prob_profit=fsum["prob_profit"])
            pid = db.save_portfolio(meta, picks)
            st.success(f"Saved as portfolio #{pid}. Open the **Compare Saved** "
                       "tab to line it up against others.")
    with save_col2:
        st.caption("Saved portfolios are stored in a SQLite database "
                   "(`portfolio.db`) so you can compare different risk levels, "
                   "amounts, and horizons side by side later.")

# ---- Risk Analysis ----------------------------------------------------
with t_risk:
    st.subheader("Risk analysis")
    st.write("Based on how this exact basket of stocks behaved over the last ~2 years.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual volatility", f"{metrics['annual_vol']:.1%}",
              help="How much the portfolio's value swings per year. Higher = riskier.")
    c2.metric("Historical annual return", f"{metrics['annual_return']:+.1%}",
              help="Average yearly return over the lookback window (past, not a promise).")
    c3.metric("Sharpe ratio", f"{metrics['sharpe']:.2f}",
              help="Return earned per unit of risk. Higher is better; >1 is good.")
    c4.metric("Worst drawdown", f"{metrics['max_drawdown']:.1%}",
              help="Biggest peak-to-trough fall seen in the lookback window.")

    if realized_band == risk:
        st.success(f"Realised risk band: **{realized_band}** — matches your "
                   f"chosen appetite. {realized_note}")
    else:
        st.warning(f"You chose **{risk}** risk, but this basket's realised "
                   f"volatility reads as **{realized_band}**. {realized_note} "
                   "That can happen when the model's best picks cluster in a "
                   "particular type of stock.")

    st.caption("Historical value of this portfolio (had you invested at the start of the window)")
    curve = (1 + port_ret).cumprod() * amount
    st.line_chart(curve.rename("Portfolio value (USD)"))

# ---- Prediction -------------------------------------------------------
with t_pred:
    st.subheader(f"Forecast: where ${amount:,.0f} could be after {horizon} trading days")
    st.write("Built from **5,000 Monte Carlo simulations** that resample this "
             "portfolio's real daily returns. The spread shows the range of "
             "plausible outcomes — not a single guaranteed number.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected value", f"${fsum['expected_value']:,.0f}",
              f"{fsum['expected_return']:+.1%}")
    c2.metric("Best case (95%)", f"${fsum['best_case']:,.0f}")
    c3.metric("Worst case (5%)", f"${fsum['worst_case']:,.0f}")
    c4.metric("Chance of profit", f"{fsum['prob_profit']:.0%}")

    fan = bands[["p5", "p50", "p95"]].copy()
    fan.columns = ["Worst case (5%)", "Expected (median)", "Best case (95%)"]
    st.caption("Projected value over time")
    st.line_chart(fan)

    st.caption("Distribution of simulated final values")
    counts, edges = np.histogram(final_values, bins=40)
    hist = pd.DataFrame({"Final value": ((edges[:-1] + edges[1:]) / 2).round(0),
                         "Simulations": counts}).set_index("Final value")
    st.bar_chart(hist)

# ---- Conclusion -------------------------------------------------------
with t_conc:
    st.subheader("Conclusion")
    direction = "grow" if fsum["expected_return"] >= 0 else "decline"
    top3 = ", ".join(picks["Ticker"].head(3))
    confidence = ("a majority of simulations end in profit"
                  if fsum["prob_profit"] >= 0.5
                  else "fewer than half of simulations end in profit")

    st.markdown(f"""
For an investment of **\\${amount:,.0f}** at a **{risk}** risk appetite, the system
selects **{len(picks)} stocks** (led by **{top3}**) chosen by the machine-learning
model and weighted to match your risk level.

This basket's realised volatility reads as **{realized_band}**, with a historical
Sharpe ratio of **{metrics['sharpe']:.2f}** and a worst observed drawdown of
**{metrics['max_drawdown']:.1%}**.

Over the next **{horizon} trading days**, the Monte Carlo forecast expects the
investment to **{direction}** to about **\\${fsum['expected_value']:,.0f}**
(**{fsum['expected_return']:+.1%}**), with a plausible range of
**\\${fsum['worst_case']:,.0f}** to **\\${fsum['best_case']:,.0f}**. On these
simulations, **{confidence}** ({fsum['prob_profit']:.0%} chance of profit).
""")

    if fsum["prob_profit"] >= 0.55:
        st.success("Overall: a reasonably favourable setup on these assumptions.")
    elif fsum["prob_profit"] >= 0.45:
        st.info("Overall: roughly balanced odds — gains and losses are similarly likely.")
    else:
        st.warning("Overall: the odds lean unfavourable over this horizon — "
                   "consider a longer horizon or a lower-risk profile.")

    st.caption("⚠️ This is an academic prototype, not financial advice. Forecasts are "
               "statistical simulations of past behaviour and do not guarantee future "
               "results. Real investing also involves fees, taxes, and live market risk.")

# ---- Compare Saved ----------------------------------------------------
with t_comp:
    st.subheader("Compare saved portfolios")
    saved = db.list_portfolios()

    if saved.empty:
        st.info("No saved portfolios yet. Build one, then click "
                "**💾 Save this portfolio** on the Recommendations tab. "
                "Try saving a Low, a Medium, and a High to compare them here.")
    else:
        st.caption(f"{len(saved)} portfolio(s) saved in the SQLite database.")
        options = {f"#{r.id} · {r.label}  ({r.created_at})": r.id
                   for r in saved.itertuples()}
        chosen = st.multiselect("Pick portfolios to compare",
                                list(options.keys()),
                                default=list(options.keys())[:min(3, len(options))])
        ids = [options[c] for c in chosen]

        if ids:
            comp = saved[saved["id"].isin(ids)].copy()

            # side-by-side metrics table
            show = comp[["id", "label", "risk", "amount", "horizon", "n_stocks",
                         "annual_vol", "sharpe", "max_drawdown",
                         "expected_return", "prob_profit"]].copy()
            show.columns = ["ID", "Label", "Risk", "Amount", "Horizon", "Stocks",
                            "Ann. Vol", "Sharpe", "Max DD",
                            "Exp. Return", "P(profit)"]
            show["Amount"] = show["Amount"].map("${:,.0f}".format)
            show["Ann. Vol"] = show["Ann. Vol"].map("{:.1%}".format)
            show["Max DD"] = show["Max DD"].map("{:.1%}".format)
            show["Exp. Return"] = show["Exp. Return"].map("{:+.1%}".format)
            show["P(profit)"] = show["P(profit)"].map("{:.0%}".format)
            show["Sharpe"] = show["Sharpe"].map("{:.2f}".format)
            st.dataframe(show, hide_index=True, width='stretch')

            # visual comparisons
            cc1, cc2 = st.columns(2)
            with cc1:
                st.caption("Expected return by portfolio")
                st.bar_chart(comp.set_index("label")["expected_return"])
            with cc2:
                st.caption("Risk (annual volatility) by portfolio")
                st.bar_chart(comp.set_index("label")["annual_vol"])

            st.caption("Chance of profit by portfolio")
            st.bar_chart(comp.set_index("label")["prob_profit"])

            # drill into holdings of one saved portfolio
            st.markdown("---")
            drill_label = st.selectbox("Inspect the holdings of one portfolio",
                                       [c for c in chosen])
            drill_id = options[drill_label]
            h = db.get_holdings(drill_id)
            h["weight"] = h["weight"].map("{:.1%}".format)
            h["allocation"] = h["allocation"].map("${:,.0f}".format)
            h["price"] = h["price"].map("${:.2f}".format)
            h["score"] = h["score"].map("{:.3f}".format)
            st.dataframe(h, hide_index=True, width='stretch')

        st.markdown("---")
        if st.button("🗑️ Clear all saved portfolios"):
            db.clear_all()
            st.rerun()

# ---- Stock Lookup -----------------------------------------------------
with t_look:
    st.subheader("Look up a single stock")
    ticker = st.selectbox("Choose a ticker", sorted(scores["Ticker"].unique()))
    row = scores[scores["Ticker"] == ticker].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Model score", f"{row['score']:.3f}")
    c2.metric("Rank", f"{int(row['rank'])} of {len(scores)}")
    c3.metric("Last price", f"${row['Adj Close']:.2f}")
    hist = featured[featured["Ticker"] == ticker].set_index("Date")["Adj Close"].tail(252)
    st.caption(f"{ticker} price — last ~1 year")
    st.line_chart(hist)

# ---- Backtest ---------------------------------------------------------
with t_back:
    st.subheader("Does the model actually add value?")
    st.write("Test-period predictions sorted into 10 buckets by score. The top "
             "bucket (the model's favourites) should out-earn the bottom bucket.")
    decile_returns, market = get_backtest()
    chart_df = pd.DataFrame({"Avg forward return": decile_returns.values},
                            index=[f"D{int(d)}" for d in decile_returns.index])
    st.bar_chart(chart_df)
    top, bottom = decile_returns.iloc[-1], decile_returns.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top decile", f"{top:+.2%}")
    c2.metric("Market avg", f"{market:+.2%}")
    c3.metric("Bottom decile", f"{bottom:+.2%}")
    c4.metric("Long-short edge", f"{top - bottom:+.2%}")
    if os.path.exists(config.PLOT_OUT):
        st.image(config.PLOT_OUT, width='stretch')
