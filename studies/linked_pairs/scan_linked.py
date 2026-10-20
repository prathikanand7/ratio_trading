"""
Pre-registered test of economically linked pairs over ~6 years of Binance data.

The pair list and rationale below were fixed BEFORE looking at any result, so
the multiple-testing count is the number of pairs listed here, not a scan.

Design
  - Data: Binance Vision 1h, last 72 complete months (2020-10 -> 2026-09).
  - DEV period: everything before 2024-10-01. Cointegration and walk-forward
    run here only.
  - HOLDOUT: 2024-10-01 onward. Not touched by this script. (Caveat: that
    window was already used for ETH/BTC and the 10-coin scan, so it is clean
    for these pairs but not a pristine holdout for the coins involved.)
"""

import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from core.data import fetch_vision
from studies.eth_btc.walkforward import walk_forward

MONTHS = 72
HOLDOUT_START = pd.Timestamp("2024-10-01", tz="UTC")

# (leg A, leg B, economic rationale)
PAIRS = [
    ("LTCUSDT", "BCHUSDT", "Old PoW payment coins, both Bitcoin-derived codebases"),
    ("XLMUSDT", "XRPUSDT", "Cross-border payment networks, shared narrative and buyers"),
    ("SOLUSDT", "AVAXUSDT", "Competing high-throughput L1s, same sector flows"),
    ("UNIUSDT", "AAVEUSDT", "DeFi blue chips, driven by the same DeFi activity cycle"),
    ("ETCUSDT", "ETHUSDT", "Common origin and codebase, shared smart-contract narrative"),
]
REFERENCE = ("ETHUSDT", "BTCUSDT", "Reference: the rejected pair, same dev window")


def annualised_sr(r):
    return r.mean() / r.std() * np.sqrt(24 * 365) if r.std() > 0 else float("nan")


def main():
    syms = sorted({s for a, b, _ in PAIRS + [REFERENCE] for s in (a, b)})
    closes = {}
    for s in syms:
        df = fetch_vision(s, "1h", MONTHS)
        closes[s] = df["close"]
        print(f"{s}: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
    px = pd.DataFrame(closes)

    alpha = 0.05 / len(PAIRS)
    print(f"\nPairs tested: {len(PAIRS)}  Bonferroni alpha = {alpha:.3f}\n")

    for a, b, why in PAIRS + [REFERENCE]:
        pair = px[[a, b]].dropna()
        dev = pair[pair.index < HOLDOUT_START]
        n_gaps = int((dev.index.to_series().diff() != pd.Timedelta(hours=1)).sum()) - 1
        label = f"{a[:-4]}/{b[:-4]}"
        print("=" * 72)
        print(f"{label}  ({why})")
        print(f"dev bars {len(dev)}, {dev.index[0].date()} -> {dev.index[-1].date()}, "
              f"timestamp gaps: {n_gaps}")

        half = np.log(dev.iloc[: len(dev) // 2])
        p_half = coint(half[a], half[b])[1]
        p_dev = coint(np.log(dev[a]), np.log(dev[b]))[1]
        print(f"Engle-Granger p: first half {p_half:.3f}, full dev {p_dev:.3f}  "
              f"(threshold {alpha:.3f})")

        if len(dev) < 24 * 30 * 8 + 1:
            print("too little history for walk-forward, skipped")
            continue
        df, oos = walk_forward(dev[a], dev[b])
        print(f"walk-forward: {len(df)} windows, profitable "
              f"{(df['test_net'] > 0).sum()}/{len(df)}, "
              f"median SR {df['test_SR'].median():.2f}, "
              f"stitched OOS {oos.sum():+.2%}, annualised SR {annualised_sr(oos):.2f}")
        by_year = oos.groupby(oos.index.year).sum()
        print("by year: " + ", ".join(f"{y}: {v:+.1%}" for y, v in by_year.items()))


if __name__ == "__main__":
    sys.exit(main())
