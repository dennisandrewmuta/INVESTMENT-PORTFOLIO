"""
database.py
-----------
A lightweight SQLite persistence layer so the system can REMEMBER portfolios
across runs and COMPARE them later.

Why SQLite (and not PostgreSQL/MySQL)?
  * Zero setup — it's a single file (portfolio.db), no server to install or run.
  * Built into Python's standard library (import sqlite3), so no extra dependency.
  * More than enough for a single-user analytical prototype at this scale.
  * You'd only move to PostgreSQL/MySQL for a multi-user, deployed product.

Schema (a proper relational design with a foreign key):

    portfolios (1) ---- (many) holdings

  portfolios : one row per saved portfolio run (the inputs + the headline results)
  holdings   : one row per stock inside a saved portfolio (linked by portfolio_id)
"""

import sqlite3
from datetime import datetime
import pandas as pd
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT,
    created_at      TEXT NOT NULL,
    amount          REAL,
    risk            TEXT,
    horizon         INTEGER,
    n_stocks        INTEGER,
    invested        REAL,
    annual_vol      REAL,
    annual_return   REAL,
    sharpe          REAL,
    max_drawdown    REAL,
    expected_value  REAL,
    expected_return REAL,
    worst_case      REAL,
    best_case       REAL,
    prob_profit     REAL
);

CREATE TABLE IF NOT EXISTS holdings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    ticker       TEXT,
    weight       REAL,
    allocation   REAL,
    shares       INTEGER,
    price        REAL,
    score        REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
);
"""


def get_conn():
    """Open a connection with foreign-key enforcement turned on."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Create the tables if they don't already exist. Safe to call every run."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def save_portfolio(meta, picks):
    """
    Insert one portfolio and all its holdings in a single transaction.

    meta  : dict with the input + result fields (see keys below)
    picks : DataFrame with Ticker, weight, allocation, shares, Adj Close, score
    Returns the new portfolio's id.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO portfolios
               (label, created_at, amount, risk, horizon, n_stocks, invested,
                annual_vol, annual_return, sharpe, max_drawdown,
                expected_value, expected_return, worst_case, best_case, prob_profit)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                meta.get("label") or f"{meta['risk']} · ${meta['amount']:,.0f} · {meta['horizon']}d",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                meta["amount"], meta["risk"], meta["horizon"], meta["n_stocks"],
                meta["invested"], meta["annual_vol"], meta["annual_return"],
                meta["sharpe"], meta["max_drawdown"], meta["expected_value"],
                meta["expected_return"], meta["worst_case"], meta["best_case"],
                meta["prob_profit"],
            ),
        )
        pid = cur.lastrowid

        rows = [
            (pid, r["Ticker"], float(r["weight"]), float(r["allocation"]),
             int(r["shares"]), float(r["Adj Close"]), float(r["score"]))
            for _, r in picks.iterrows()
        ]
        cur.executemany(
            """INSERT INTO holdings
               (portfolio_id, ticker, weight, allocation, shares, price, score)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        return pid


def list_portfolios():
    """Return all saved portfolios as a DataFrame (newest first)."""
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM portfolios ORDER BY id DESC", conn)


def get_holdings(portfolio_id):
    """Return the holdings for one saved portfolio."""
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT ticker, weight, allocation, shares, price, score "
            "FROM holdings WHERE portfolio_id = ? ORDER BY weight DESC",
            conn, params=(int(portfolio_id),))


def delete_portfolio(portfolio_id):
    """Delete one saved portfolio (its holdings go too, via ON DELETE CASCADE)."""
    with get_conn() as conn:
        conn.execute("DELETE FROM portfolios WHERE id = ?", (int(portfolio_id),))
        conn.commit()


def clear_all():
    """Wipe every saved portfolio. Handy for resetting before a fresh demo."""
    with get_conn() as conn:
        conn.execute("DELETE FROM holdings")
        conn.execute("DELETE FROM portfolios")
        conn.commit()


if __name__ == "__main__":
    # Quick self-test
    init_db()
    import pandas as pd
    demo_meta = dict(label="TEST", amount=10000, risk="Low", horizon=60,
                     n_stocks=2, invested=9800, annual_vol=0.16, annual_return=0.12,
                     sharpe=0.75, max_drawdown=-0.2, expected_value=10260,
                     expected_return=0.026, worst_case=8996, best_case=11716,
                     prob_profit=0.64)
    demo_picks = pd.DataFrame({
        "Ticker": ["AAA", "BBB"], "weight": [0.6, 0.4],
        "allocation": [6000, 4000], "shares": [10, 8],
        "Adj Close": [600.0, 500.0], "score": [0.58, 0.55]})
    pid = save_portfolio(demo_meta, demo_picks)
    print("saved portfolio id:", pid)
    print(list_portfolios()[["id", "label", "amount", "risk"]].to_string(index=False))
    print(get_holdings(pid).to_string(index=False))
    clear_all()
    print("cleared. remaining:", len(list_portfolios()))
