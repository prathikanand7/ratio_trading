"""
Data loading for the ETH/BTC ratio mean reversion study.

Three sources, tried in order:

  1. Binance Vision (data.binance.vision) -- a free, unauthenticated static
     file host holding monthly zipped CSVs of every kline Binance has ever
     published. No API key, no account, no rate limit. This is the right way
     to pull bulk history.
  2. Binance REST (api.binance.com) -- live endpoint, capped at 1000 bars per
     call, so multi-year history means paging. Useful for topping up the
     current month, which Vision has not published yet.
  3. Synthetic -- a generated cointegrated pair, so the pipeline can be
     validated with no network at all. Clearly labelled as not real.
"""

import io
import os
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import urllib.request
import json

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_VISION = "https://data.binance.vision"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base", "taker_quote", "ignore",
]


def _months_back(n_months):
    """
    (year, month) pairs for the last n complete months.

    The current month is skipped: Vision publishes a monthly archive only once
    the month has ended, so asking for it returns a 404.
    """
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    out = []
    for _ in range(n_months):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def _parse_kline_zip(blob):
    """
    Parse one Binance Vision monthly archive into a DataFrame.

    Two quirks this handles:
      - Older archives have no header row, newer ones do. Sniffed rather than
        assumed, because getting it wrong silently turns the header into a
        data row and poisons the first bar.
      - Binance moved timestamps from milliseconds to microseconds partway
        through the archive. Detected by magnitude.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            head = fh.read(64).decode("utf-8", "ignore")
        has_header = "open_time" in head.lower()
        with z.open(name) as fh:
            df = pd.read_csv(
                fh,
                header=0 if has_header else None,
                names=None if has_header else KLINE_COLS,
            )

    df.columns = KLINE_COLS[:len(df.columns)]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ct = pd.to_numeric(df["close_time"], errors="coerce")
    unit = "us" if ct.max() > 1e15 else "ms"
    df["ts"] = pd.to_datetime(ct, unit=unit, utc=True)

    return df[["ts", "open", "high", "low", "close", "volume"]].dropna(subset=["ts"])


def fetch_vision(symbol, interval="1h", months=24, market="spot"):
    """
    Pull monthly kline archives from Binance Vision.

    URL shape:
      <base>/data/spot/monthly/klines/ETHUSDT/1h/ETHUSDT-1h-2025-03.zip

    A missing month is reported rather than silently skipped, because a gap in
    the middle of the sample quietly breaks the rolling statistics.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"vision_{symbol}_{interval}_{months}m.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    frames, missing = [], []

    for y, m in _months_back(months):
        url = (f"{BINANCE_VISION}/data/{market}/monthly/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{y}-{m:02d}.zip")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
        except Exception as e:
            missing.append(f"{y}-{m:02d} ({type(e).__name__})")
            continue
        frames.append(_parse_kline_zip(blob))

    if not frames:
        raise RuntimeError(
            f"Binance Vision returned nothing for {symbol}. "
            f"First few failures: {missing[:5]}"
        )
    if missing:
        print(f"[data] {symbol}: {len(missing)} month(s) unavailable: {missing[:3]}")

    df = pd.concat(frames, ignore_index=True).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(cache)
    return df


def _fetch_klines_page(symbol, interval, start_ms, limit=1000):
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch_binance(symbol, interval="1h", days=720):
    """Page through Binance klines. Returns DataFrame indexed by close time (UTC)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{days}d.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    rows = []
    cursor = start_ms

    while cursor < end_ms:
        page = _fetch_klines_page(symbol, interval, cursor)
        if not page:
            break
        rows.extend(page)
        cursor = page[-1][6] + 1  # close time of last bar + 1ms
        if len(page) < 1000:
            break
        time.sleep(0.25)  # be polite to the public endpoint

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["ts", "open", "high", "low", "close", "volume"]].set_index("ts")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_parquet(cache)
    return df


def synthetic_pair(n=17_520, seed=7):
    """
    Generate a synthetic cointegrated pair for pipeline validation.

    Construction: BTC is a random walk. ETH follows BTC with a beta, plus an
    OU (mean-reverting) residual. This gives a spread that genuinely reverts,
    so a correct backtest SHOULD find a signal here. If the backtest finds
    nothing on this data, the backtest is broken.
    """
    rng = np.random.default_rng(seed)

    # BTC: geometric random walk
    btc_ret = rng.normal(0, 0.004, n)
    btc = 30_000 * np.exp(np.cumsum(btc_ret))

    # OU residual: dx = -theta*x*dt + sigma*dW
    theta, sigma = 0.02, 0.01
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = resid[i - 1] - theta * resid[i - 1] + sigma * rng.normal()

    beta = 1.0
    log_eth = np.log(0.06) + beta * np.log(btc) + resid
    eth = np.exp(log_eth)

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return (
        pd.DataFrame({"close": btc}, index=idx),
        pd.DataFrame({"close": eth}, index=idx),
    )


def load_pair(sym_a="ETHUSDT", sym_b="BTCUSDT", interval="1h", days=720,
              allow_synthetic=True, source="auto"):
    """
    Returns (df_a, df_b, source_used), aligned on their common timestamps.

    source: "auto" (Vision, then REST, then synthetic), "vision", "rest",
    or "synthetic".

    The inner join on timestamps matters. If one symbol is missing an hour
    that the other has, keeping it would mean computing a spread from a stale
    price on one leg, which shows up as a fake deviation and a fake trade.
    """
    months = max(1, round(days / 30.44))
    attempts = []

    if source in ("auto", "vision"):
        attempts.append(("vision", lambda s: fetch_vision(s, interval, months)))
    if source in ("auto", "rest"):
        attempts.append(("rest", lambda s: fetch_binance(s, interval, days)))

    for name, fn in attempts:
        try:
            a, b = fn(sym_a), fn(sym_b)
            idx = a.index.intersection(b.index)
            if len(idx) == 0:
                raise RuntimeError("no overlapping timestamps between the two symbols")
            dropped = (len(a) - len(idx)) + (len(b) - len(idx))
            if dropped:
                print(f"[data] dropped {dropped} unmatched bar(s) on the inner join")
            return a.loc[idx], b.loc[idx], name
        except Exception as e:
            print(f"[data] {name} source failed ({type(e).__name__}: {e})")

    if source == "synthetic" or allow_synthetic:
        print("[data] Falling back to SYNTHETIC data. Results are not real.")
        b, a = synthetic_pair()
        return a, b, "synthetic"

    raise RuntimeError("no data source available")
