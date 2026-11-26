"""
web_db.py
---------
Database layer for the deployed website. Unlike the prototype's database.py
(plain SQLite, single user), this one:

  * works on BOTH SQLite (local dev) and PostgreSQL (production) via SQLAlchemy
  * isolates each visitor's saved portfolios by an anonymous session id
    (a signed cookie — no login, no personal data)

Which database is used is decided by the DATABASE_URL environment variable:
  * not set            -> SQLite file (portfolio_web.db)  [local development]
  * set (postgres://…) -> that PostgreSQL database          [production]

So the same code runs locally and in the cloud with zero changes.
"""

import os
import datetime
from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, String,
                        Float, DateTime, ForeignKey, select, insert, delete)

# ---- pick the database from the environment ----
SQLITE_URL = "sqlite:///portfolio_web.db"
url = os.environ.get("DATABASE_URL", "").strip() or SQLITE_URL
if url.startswith("postgres://"):
    url = url.replace("postgres://", "postgresql://", 1)


def _make_engine(u):
    return create_engine(
        u, pool_pre_ping=True,
        connect_args={"check_same_thread": False} if u.startswith("sqlite") else {})


try:
    engine = _make_engine(url)
    with engine.connect():
        pass
except Exception as exc:  # missing driver, dead database, bad URL...
    print("Database unavailable (%s) - falling back to SQLite" % exc)
    engine = _make_engine(SQLITE_URL)
meta = MetaData()

portfolios = Table(
    "web_portfolios", meta,
    Column("id", Integer, primary_key=True),
    Column("session_id", String(64), index=True),
    Column("label", String(120)),
    Column("created_at", DateTime, default=datetime.datetime.utcnow),
    Column("amount", Float), Column("risk", String(10)), Column("horizon", Integer),
    Column("n_stocks", Integer), Column("invested", Float),
    Column("annual_vol", Float), Column("annual_return", Float),
    Column("sharpe", Float), Column("max_drawdown", Float),
    Column("expected_value", Float), Column("expected_return", Float),
    Column("worst_case", Float), Column("best_case", Float), Column("prob_profit", Float),
)

holdings = Table(
    "web_holdings", meta,
    Column("id", Integer, primary_key=True),
    Column("portfolio_id", Integer, ForeignKey("web_portfolios.id", ondelete="CASCADE")),
    Column("ticker", String(12)), Column("weight", Float), Column("allocation", Float),
    Column("shares", Integer), Column("price", Float), Column("score", Float),
)


def init_db():
    meta.create_all(engine)


def save_portfolio(session_id, m, picks):
    """Insert one portfolio + its holdings for a given session."""
    with engine.begin() as conn:
        pid = conn.execute(insert(portfolios).values(
            session_id=session_id, label=m["label"], amount=m["amount"], risk=m["risk"],
            horizon=m["horizon"], n_stocks=m["n_stocks"], invested=m["invested"],
            annual_vol=m["annual_vol"], annual_return=m["annual_return"], sharpe=m["sharpe"],
            max_drawdown=m["max_drawdown"], expected_value=m["expected_value"],
            expected_return=m["expected_return"], worst_case=m["worst_case"],
            best_case=m["best_case"], prob_profit=m["prob_profit"],
        )).inserted_primary_key[0]
        for h in picks:
            conn.execute(insert(holdings).values(
                portfolio_id=pid, ticker=h["ticker"], weight=h["weight"],
                allocation=h["allocation"], shares=h["shares"], price=h["price"], score=h["score"]))
        return pid


def list_portfolios(session_id):
    """Return this session's saved portfolios, newest first."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(portfolios).where(portfolios.c.session_id == session_id)
            .order_by(portfolios.c.id.desc())
        ).mappings().all()
    return [dict(r) for r in rows]


def clear_session(session_id):
    """Delete this session's saved portfolios (and their holdings)."""
    with engine.begin() as conn:
        ids = [r[0] for r in conn.execute(
            select(portfolios.c.id).where(portfolios.c.session_id == session_id)).all()]
        if ids:
            conn.execute(delete(holdings).where(holdings.c.portfolio_id.in_(ids)))
            conn.execute(delete(portfolios).where(portfolios.c.id.in_(ids)))
