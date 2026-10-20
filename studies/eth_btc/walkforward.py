"""
Walk-forward test: repeatedly fit on a trailing train window, trade the next
test window with the parameters and hedge ratio chosen on train only, then
roll forward. Stitching the test windows together gives an out-of-sample
record that is not one lucky split.
"""

import numpy as np
import pandas as pd

from core.data import load_pair
from core.backtest import Params, run_backtest, trade_list, estimate_beta
from core.metrics import summarize
from studies.eth_btc.run_study import search

TRAIN_BARS = 24 * 30 * 6   # ~6 months
TEST_BARS = 24 * 30 * 2    # ~2 months
GRID = {
    "lookback": [72, 168, 336],
    "entry_z": [1.5, 2.0, 2.5],
    "exit_z": [0.0, 0.5, 1.0],
    "max_hold": [120, 240],
}


def walk_forward(px_a, px_b):
    """Returns (per-window DataFrame, stitched out-of-sample net returns)."""
    rows, oos = [], []
    start = 0
    while start + TRAIN_BARS + TEST_BARS <= len(px_a):
        tr = slice(start, start + TRAIN_BARS)
        te = slice(start + TRAIN_BARS, start + TRAIN_BARS + TEST_BARS)
        a_tr, b_tr = px_a.iloc[tr], px_b.iloc[tr]
        a_te, b_te = px_a.iloc[te], px_b.iloc[te]

        beta = estimate_beta(a_tr, b_tr)
        base = Params(beta_mode="fixed", fixed_beta=beta)
        ranked = search(a_tr, b_tr, base, GRID, min_trades=8)
        best = ranked.iloc[0]
        chosen = Params(**{**base.__dict__, **best["params"]})

        f = run_backtest(a_te, b_te, chosen)
        s = summarize(f, trade_list(f), "test")
        oos.append(f["net_ret"].fillna(0))
        rows.append({
            "test_start": a_te.index[0].date(), "beta": round(beta, 3),
            "train_SR": round(best["sharpe_net"], 2),
            "test_SR": round(s["sharpe_net"], 2),
            "test_net": s["total_net"], "trades": s["trades"],
            "params": best["params"],
        })
        start += TEST_BARS

    return pd.DataFrame(rows), pd.concat(oos)


def main():
    a, b, source = load_pair(source="vision", allow_synthetic=False)
    px_a, px_b = a["close"], b["close"]
    print(f"Data: {source}, {len(px_a)} bars, {px_a.index[0].date()} -> {px_a.index[-1].date()}\n")

    df, r = walk_forward(px_a, px_b)
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    print(df.to_string(index=False, formatters={"test_net": "{:+.2%}".format}))

    sr =r.mean() / r.std() * np.sqrt(24 * 365)
    print(f"\nWindows: {len(df)}, profitable: {(df['test_net'] > 0).sum()}/{len(df)}")
    print(f"Stitched OOS net return: {r.sum():+.2%}, annualised Sharpe: {sr:.2f}")
    print(f"Median window test SR: {df['test_SR'].median():.2f}")


if __name__ == "__main__":
    main()
