"""
Scan many pairs for cointegration on the TRAIN window only, then walk-forward
test the ones that pass.

With N pairs tested at the 5% level, about 0.05*N will pass by luck alone, so
the cointegration cut is Bonferroni-adjusted and the walk-forward result is the
real filter.
"""

import itertools
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from core.data import fetch_vision
from studies.eth_btc.walkforward import walk_forward

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT",
           "AVAXUSDT", "LTCUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
TRAIN_FRAC = 0.7


def main():
    closes = {}
    for s in SYMBOLS:
        try:
            closes[s] = fetch_vision(s, "1h", 24)["close"]
        except Exception as e:
            print(f"[skip] {s}: {e}")
    px = pd.DataFrame(closes).dropna()
    print(f"{px.shape[1]} symbols, {len(px)} common bars, "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")

    cut = int(len(px) * TRAIN_FRAC)
    train = np.log(px.iloc[:cut])
    pairs = list(itertools.combinations(px.columns, 2))
    alpha = 0.05 / len(pairs)

    rows = []
    for a, b in pairs:
        p = coint(train[a], train[b])[1]
        rows.append({"pair": f"{a[:-4]}/{b[:-4]}", "a": a, "b": b, "p": p})
    res = pd.DataFrame(rows).sort_values("p")
    print(f"Engle-Granger on train only, {len(pairs)} pairs, "
          f"Bonferroni alpha = {alpha:.4f}")
    print(res.head(10)[["pair", "p"]].to_string(index=False, float_format="{:.4f}".format))

    passers = res[res["p"] < alpha]
    if passers.empty:
        print("\nNo pair is cointegrated after multiple-testing correction.")
        passers = res.head(3)
        print("Walk-forward on the 3 lowest p-values anyway, for reference:")
    else:
        print(f"\n{len(passers)} pair(s) pass the corrected threshold.")

    for _, r in passers.iterrows():
        df, oos = walk_forward(px[r["a"]], px[r["b"]])
        sr = oos.mean() / oos.std() * np.sqrt(24 * 365)
        print(f"\n{r['pair']}  (coint p={r['p']:.4f})")
        print(f"  profitable windows {(df['test_net'] > 0).sum()}/{len(df)}, "
              f"median SR {df['test_SR'].median():.2f}, "
              f"stitched OOS {oos.sum():+.2%}, annualised SR {sr:.2f}")


if __name__ == "__main__":
    sys.exit(main())
