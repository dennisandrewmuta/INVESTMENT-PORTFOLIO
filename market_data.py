"""
market_data.py
--------------
Live market data for the website: index levels (S&P 500, Nasdaq, ...) and the
latest price of every stock the model ranks.

Design goals
  * NEVER block or break a page. Fetching happens in a background thread; if the
    data provider is slow, rate-limited or down, the site keeps working with the
    saved data bundle and says so ("Live feed unavailable").
  * Be polite to the (free) provider. Results are cached in memory and in a small
    file, and refreshed only when stale. Refresh is much slower when the US
    market is closed (prices can't change).
  * Swappable provider. Only fetch_indices() and fetch_prices() talk to Yahoo
    Finance; to use another API later, replace those two functions.

Settings (optional environment variables)
  LIVE_DATA=0               turn live data off completely
  MARKET_TTL_INDICES=300    seconds between index refreshes while market is open
  MARKET_TTL_PRICES=1800    seconds between stock-price refreshes while open
"""

import os
import json
import time
import math
import tempfile
import threading
from datetime import datetime, timezone

import pandas as pd

LIVE_ENABLED = os.environ.get("LIVE_DATA", "1") != "0"
TTL_INDICES = int(os.environ.get("MARKET_TTL_INDICES", "300"))
TTL_PRICES = int(os.environ.get("MARKET_TTL_PRICES", "1800"))
TTL_CLOSED = 6 * 3600          # refresh interval while the market is closed
RETRY_MIN, RETRY_MAX = 120, 900  # back-off after a failed fetch (seconds)
BATCH = 120                    # tickers per download request
CACHE_FILE = os.path.join(tempfile.gettempdir(), "sips_market_cache.json")

INDICES = [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^DJI", "Dow Jones"),
           ("^RUT", "Russell 2000"), ("^VIX", "VIX")]

_lock = threading.Lock()
_cache = {
    "indices": [], "indices_ts": 0.0,
    "prices": {}, "prices_ts": 0.0, "prices_date": None,
    "fail_count": 0, "next_try": 0.0, "last_error": None,
    "refreshing": False, "loaded_file": False,
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def market_open(now=None):
    """True during regular US trading hours (Mon-Fri 9:30-16:00 New York).
    Ignores exchange holidays - the worst case is one extra refresh."""
    try:
        from zoneinfo import ZoneInfo
        now = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    except Exception:
        now = now or datetime.now(timezone.utc)
        now = now.replace(hour=(now.hour - 5) % 24)  # rough UTC-5 fallback
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def _ttl(base):
    return base if market_open() else max(base, TTL_CLOSED)


def _yahoo_symbol(ticker):
    """Dataset uses BRK.B; Yahoo wants BRK-B."""
    return str(ticker).replace(".", "-")


def _good(x):
    try:
        return x is not None and math.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError):
        return False


def _close_frame(df, symbols):
    """Return a DataFrame of closing prices (columns = symbols) from whatever
    shape yfinance gives back (multi-level in either order, or single ticker)."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        for level in range(cols.nlevels):
            if "Close" in cols.get_level_values(level):
                out = df.xs("Close", axis=1, level=level)
                return out if isinstance(out, pd.DataFrame) else out.to_frame(symbols[0])
        return pd.DataFrame()
    if "Close" in cols:
        return df[["Close"]].rename(columns={"Close": symbols[0]})
    return pd.DataFrame()


def _download(symbols, period):
    import yfinance as yf
    return yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                       progress=False, threads=True, timeout=20)


# ----------------------------------------------------------------------------
# provider: Yahoo Finance (replace these two to switch data source)
# ----------------------------------------------------------------------------
def fetch_indices():
    """List of {symbol, name, price, change, pct, spark[]} for the major indices."""
    symbols = [s for s, _ in INDICES]
    closes = _close_frame(_download(symbols, "1mo"), symbols)
    out = []
    for sym, name in INDICES:
        if sym not in closes:
            continue
        s = closes[sym].dropna()
        if len(s) < 2 or not _good(s.iloc[-1]):
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        out.append({"symbol": sym, "name": name, "price": round(last, 2),
                    "change": round(last - prev, 2),
                    "pct": round((last / prev - 1) * 100, 2) if prev else 0.0,
                    "spark": [round(float(v), 2) for v in s.tail(22).values]})
    return out


def fetch_prices(tickers):
    """{ticker: latest_close} for the given dataset tickers + the date of the data."""
    prices, newest = {}, None
    tickers = list(tickers)
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        ymap = {_yahoo_symbol(t): t for t in chunk}
        symbols = list(ymap)
        closes = _close_frame(_download(symbols, "5d"), symbols)
        for sym, t in ymap.items():
            if sym not in closes:
                continue
            s = closes[sym].dropna()
            if len(s) and _good(s.iloc[-1]):
                prices[t] = round(float(s.iloc[-1]), 4)
                d = s.index[-1]
                newest = d if newest is None or d > newest else newest
        time.sleep(0.5)  # be gentle between batches
    date = pd.Timestamp(newest).strftime("%Y-%m-%d") if newest is not None else None
    return prices, date


# ----------------------------------------------------------------------------
# cache (memory + file, so a restart or a second worker does not start cold)
# ----------------------------------------------------------------------------
def _load_file():
    if _cache["loaded_file"]:
        return
    _cache["loaded_file"] = True
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            d = json.load(f)
        for k in ("indices", "indices_ts", "prices", "prices_ts", "prices_date"):
            if k in d:
                _cache[k] = d[k]
    except Exception:
        pass


def _save_file():
    try:
        tmp = CACHE_FILE + ".tmp"
        keys = ("indices", "indices_ts", "prices", "prices_ts", "prices_date")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: _cache[k] for k in keys}, f)
        os.replace(tmp, CACHE_FILE)
    except Exception:
        pass


def _refresh(tickers, want_indices, want_prices):
    """Runs in a background thread. Updates the cache; never raises."""
    try:
        if want_indices:
            data = fetch_indices()
            if data:
                with _lock:
                    _cache["indices"], _cache["indices_ts"] = data, time.time()
        if want_prices and tickers:
            prices, date = fetch_prices(tickers)
            if prices:
                with _lock:
                    _cache["prices"], _cache["prices_ts"] = prices, time.time()
                    _cache["prices_date"] = date
        with _lock:
            ok = (not want_indices or _cache["indices_ts"] > time.time() - 60) and \
                 (not want_prices or not tickers or _cache["prices_ts"] > time.time() - 60)
            if ok:
                _cache["fail_count"], _cache["last_error"], _cache["next_try"] = 0, None, 0.0
            else:
                raise RuntimeError("provider returned no usable data")
        _save_file()
    except Exception as exc:  # network error, rate limit, missing package...
        with _lock:
            _cache["fail_count"] += 1
            wait = min(RETRY_MAX, RETRY_MIN * 2 ** (_cache["fail_count"] - 1))
            _cache["next_try"] = time.time() + wait
            _cache["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
        print(f"[market_data] refresh failed ({_cache['last_error']}); retry in {wait}s")
    finally:
        with _lock:
            _cache["refreshing"] = False


def ensure_fresh(tickers=None):
    """Start a background refresh if the cache is stale. Returns immediately."""
    if not LIVE_ENABLED:
        return
    with _lock:
        _load_file()
        now = time.time()
        if _cache["refreshing"] or now < _cache["next_try"]:
            return
        want_i = now - _cache["indices_ts"] > _ttl(TTL_INDICES)
        want_p = bool(tickers) and now - _cache["prices_ts"] > _ttl(TTL_PRICES)
        if not (want_i or want_p):
            return
        _cache["refreshing"] = True
    threading.Thread(target=_refresh, args=(list(tickers or []), want_i, want_p),
                     daemon=True, name="market-refresh").start()


# ----------------------------------------------------------------------------
# what the rest of the app uses
# ----------------------------------------------------------------------------
def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None


def snapshot(tickers=None):
    """JSON-ready view of the market for /api/market."""
    ensure_fresh(tickers)
    with _lock:
        _load_file()
        ts = _cache["indices_ts"]
        age = time.time() - ts if ts else None
        is_open = market_open()
        if not LIVE_ENABLED:
            status = "off"
        elif not _cache["indices"]:
            status = "loading" if _cache["refreshing"] or _cache["fail_count"] == 0 else "offline"
        elif age is not None and age <= _ttl(TTL_INDICES) * 2 + 60:
            status = "live" if is_open else "closed"
        else:
            status = "stale"
        return {"status": status, "market_open": is_open, "indices": list(_cache["indices"]),
                "updated_at": _iso(ts), "age_seconds": int(age) if age is not None else None,
                "prices_date": _cache["prices_date"], "prices_count": len(_cache["prices"]),
                "error": _cache["last_error"]}


def live_prices(tickers=None):
    """({ticker: price}, info) - prices may be empty if the feed is unavailable."""
    ensure_fresh(tickers)
    with _lock:
        _load_file()
        prices = dict(_cache["prices"])
        return prices, {"date": _cache["prices_date"], "updated_at": _iso(_cache["prices_ts"]),
                        "count": len(prices)}
