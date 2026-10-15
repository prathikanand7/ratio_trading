"""
Cash-and-carry (spot long / perp short) study on Binance.

Economic anchor: a USDT perpetual is tied to spot by its funding mechanism.
When the perp trades rich, longs pay shorts every 8h, so a delta-neutral
book (long spot, short perp) collects funding and holds the basis until it
converges. The return is a carry, not a bet on mean reversion of a spread.

Rules are fixed in advance (no parameter search):
  always_on : hold the hedge the whole time.
  filtered  : hold the hedge only while the trailing 7-day mean funding rate
              (last 21 settlements) is positive. Signal uses settlements up to
              the decision bar and the position starts on the NEXT bar.

P&L per hour on 1 unit of spot notional:
    spot return - perp return          (the hedge; basis change)
  + funding rate at each 8h settlement (short receives when rate > 0)
  - trading costs on every change of position

DEV period only (before 2024-10-01) unless --holdout is passed.
"""

import argparse
import io
import os
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from core.data import fetch_vision, _months_back, _parse_kline_zip, CACHE_DIR

BASE = "https://data.binance.vision/data/futures/um/monthly"
MONTHS = 72
HOLDOUT_START = pd.Timestamp("2024-10-01", tz="UTC")
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]

# Taker fees at the base tier: spot 10 bps, USDT-M perp 4.5 bps.
# Opening or closing the hedge trades both legs.
SPOT_FEE, PERP_FEE = 0.0010, 0.00045
LEG_COST = SPOT_FEE + PERP_FEE
SLIPPAGE = 0.0002   # per leg-pair, assumed
FUND_WINDOW = 21    # 7 days of 8h settlements


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_perp(symbol, interval="1h", months=MONTHS):
    cache = os.path.join(CACHE_DIR, f"perp_{symbol}_{interval}_{months}m.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    frames, missing = [], []
    for y, m in _months_back(months):
        url = f"{BASE}/klines/{symbol}/{interval}/{symbol}-{interval}-{y}-{m:02d}.zip"
        try:
            frames.append(_parse_kline_zip(_get(url)))
        except Exception as e:
            missing.append(f"{y}-{m:02d}")
    if not frames:
        raise RuntimeError(f"no perp data for {symbol}")
    if missing:
        print(f"[basis] {symbol} perp: missing months {missing[:4]}")
    df = pd.concat(frames).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(cache)
    return df


def fetch_funding(symbol, months=MONTHS):
    cache = os.path.join(CACHE_DIR, f"funding_{symbol}_{months}m.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)["rate"]
    out, missing = [], []
    for y, m in _months_back(months):
        url = f"{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip"
        try:
            with zipfile.ZipFile(io.BytesIO(_get(url))) as z:
                out.append(pd.read_csv(z.open(z.namelist()[0])))
        except Exception:
            missing.append(f"{y}-{m:02d}")
    if missing:
        print(f"[basis] {symbol} funding: missing months {missing[:4]}")
    df = pd.concat(out)
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.round("h")
    s = df.set_index("ts")["last_funding_rate"].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="last")].rename("rate")
    s.to_frame().to_parquet(cache)
    return s


def _hourly(s):
    """Index by bar END time (close_time rounded to the hour), one row per hour."""
    s = s.copy()
    s.index = s.index.round("h")
    return s[~s.index.duplicated(keep="last")]


def build(symbol):
    """Hourly frame: spot/perp close, funding paid at settlement bars."""
    spot = fetch_vision(symbol, "1h", MONTHS)["close"]
    perp = fetch_perp(symbol)["close"]
    spot = _hourly(spot)
    perp = _hourly(perp)
    fund = fetch_funding(symbol)
    df = pd.concat({"spot": spot, "perp": perp}, axis=1).dropna()
    df["funding"] = fund.reindex(df.index).fillna(0.0)
    df["settle"] = fund.reindex(df.index).notna()
    return df


def run(df, mode):
    """Net hourly return series on 1 unit of spot notional."""
    hedge = df["spot"].pct_change() - df["perp"].pct_change()
    rate = df["funding"].where(df["settle"])
    trailing = rate.dropna().rolling(FUND_WINDOW).mean().reindex(df.index).ffill()
    if mode == "always_on":
        want = pd.Series(1.0, index=df.index)
    else:
        want = (trailing > 0).astype(float)
    pos = want.shift(1).fillna(0.0)   # decided at t, held from t+1
    gross = pos * (hedge + df["funding"])
    cost = pos.diff().abs().fillna(pos.abs()) * (LEG_COST + SLIPPAGE)
    return (gross - cost).iloc[1:], pos


def stats(r, pos):
    ann = 24 * 365
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    sr = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else float("nan")
    return {
        "ann_ret": r.mean() * ann,
        "sharpe": sr,
        "max_dd": dd,
        "in_market": pos.loc[r.index].mean(),
        "switches": int(pos.loc[r.index].diff().abs().sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="evaluate on the untouched holdout (run once)")
    args = ap.parse_args()

    for sym in ASSETS:
        df = build(sym)
        if args.holdout:
            df = df[df.index >= HOLDOUT_START]
        else:
            df = df[df.index < HOLDOUT_START]
        print("=" * 78)
        print(f"{sym}  {df.index[0].date()} -> {df.index[-1].date()}  {len(df)} bars  "
              f"mean funding/8h {df.loc[df['settle'], 'funding'].mean()*1e4:.2f} bps  "
              f"positive {100*(df.loc[df['settle'], 'funding'] > 0).mean():.0f}%")
        for mode in ("always_on", "filtered"):
            r, pos = run(df, mode)
            s = stats(r, pos)
            yearly = r.groupby(r.index.year).sum()
            print(f"  {mode:10s} ann {s['ann_ret']:+7.2%}  SR {s['sharpe']:5.2f}  "
                  f"maxDD {s['max_dd']:+7.2%}  in-mkt {s['in_market']:4.0%}  "
                  f"switches {s['switches']:4d}  | "
                  + " ".join(f"{y}:{v:+.1%}" for y, v in yearly.items()))


if __name__ == "__main__":
    sys.exit(main())
