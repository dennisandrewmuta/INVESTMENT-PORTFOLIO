"""
config.py
---------
One place to change every setting in the project. Edit values here instead of
hunting through the other files. This keeps your experiments reproducible:
when you tweak the horizon or the train/test cutoff, you change it once.
"""

# ----- Data -----
DATA_PATH = "SP500_Historical_Data.csv"   # path to the Kaggle CSV (same folder)
CACHE_PATH = "sp500_cache.pkl"            # cached cleaned data (auto-created)

# Run on a few tickers first while developing (fast). Set to None to use all 472.
# Example: SAMPLE_TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "XOM"]
SAMPLE_TICKERS = None

# ----- Prediction problem -----
# HORIZON = how many trading days ahead we look. 21 ~= one trading month.
HORIZON = 21

# We DON'T predict "will the stock go up" (the whole market drifts up, so the
# model just learns to say "up" and that's useless for picking stocks).
# We predict "will this stock BEAT the median stock over the horizon" — the
# cross-sectional signal you actually need to choose a portfolio.
# Target = 1 if the stock's forward return is above that day's median, else 0.

# ----- Train / test split -----
# Financial data MUST be split by time, never randomly. A random split lets the
# model peek at the future ("lookahead bias") and gives fake-high accuracy.
TRAIN_END_DATE = "2021-01-01"   # train on data before this, test on/after it

# ----- Features the model learns from (all computed from PAST data only) -----
FEATURES = [
    "ret_1",         # yesterday's return
    "ret_5",         # 1-week return
    "ret_21",        # 1-month return
    "px_sma20",      # price relative to its 20-day average
    "px_sma50",      # price relative to its 50-day average
    "vol_21",        # 21-day volatility (risk)
    "mom_63",        # 3-month momentum
    "volume_ratio",  # today's volume vs its 20-day average
    "rsi_14",        # Relative Strength Index (overbought/oversold)
]

# ----- Model hyperparameters (HistGradientBoosting: fast on millions of rows) -----
MODEL_PARAMS = dict(
    max_iter=300,            # number of boosting rounds (like n_estimators)
    learning_rate=0.05,      # smaller = slower but more careful learning
    max_depth=6,             # tree depth; controls model complexity
    l2_regularization=1.0,   # penalty that fights overfitting
    early_stopping=True,     # stop if validation score stops improving
    random_state=42,         # makes results reproducible
)

# ----- Output files -----
MODEL_OUT = "portfolio_model.joblib"
METRICS_OUT = "metrics.txt"
PLOT_OUT = "results.png"

# ----- Database -----
DB_PATH = "portfolio.db"   # SQLite file (auto-created)
