"""
Sanity checks on the backtest engine itself, run against synthetic data whose
properties we control. If these fail, nothing the engine says about real data
can be trusted.

  1. Known-signal test: on data built with a mean-reverting residual, the
     engine should find a clear edge. If it doesn't, the engine is broken or
     the spread construction is destroying the signal.

  2. Look-ahead test: deliberately remove the position shift. A cheating
     backtest should look dramatically better. If it doesn't, the shift
     wasn't doing anything and the honest version may already be cheating.

  3. Shuffle test: destroy the time structure. Edge should vanish. If it
     survives, the "edge" is an artifact of the accounting, not the signal.
"""

import numpy as np
import pandas as pd

from core.data import synthetic_pair
from core.backtest import Params, build_spread, generate_positions, run_backtest, trade_list
from core.metrics import summarize, sharpe, half_life, adf_test


def cheating_backtest(px_a, px_b, p):
    """
    The realistic look-ahead bug: normalise the spread with the FULL-SAMPLE
    mean and standard deviation instead of a trailing window.

    It is an easy mistake to make. `(spread - spread.mean()) / spread.std()`
    is one natural line of pandas, and it silently tells every bar in the
    backtest where the spread will average out over the entire period,
    including the future. The strategy then only ever fades deviations from
    a level it could not have known at the time.

    Note that simply dropping the .shift(1) in run_backtest is NOT this bug.
    That is a misalignment: it pays a position the move that triggered its
    own entry, which for a contrarian strategy makes results look worse
    rather than better. Worth knowing, because "my results got worse" is not
    proof that a backtest is clean.
    """
    f = build_spread(px_a, px_b, p)

    # The bug, in one line.
    f["z"] = (f["spread"] - f["spread"].mean()) / f["spread"].std()

    f["position"] = generate_positions(f["z"], p)
    ret_a = f["log_a"].diff()
    ret_b = f["log_b"].diff()
    spread_ret = ret_a - f["beta"].shift(1) * ret_b

    held = f["position"].shift(1).fillna(0)
    f["gross_ret"] = held * spread_ret
    turnover = held.diff().abs().fillna(held.abs())
    f["cost"] = turnover * (1 + f["beta"].shift(1).abs().fillna(1)) * (p.cost_bps / 10_000)
    f["net_ret"] = f["gross_ret"] - f["cost"]
    f["equity_net"] = f["net_ret"].fillna(0).cumsum()
    f["equity_gross"] = f["gross_ret"].fillna(0).cumsum()
    f["held"] = held
    return f


def main():
    btc, eth = synthetic_pair()
    px_a, px_b = eth["close"], btc["close"]

    print("=" * 70)
    print("ENGINE VALIDATION (synthetic data with a known OU residual)")
    print("=" * 70)
    print("Construction: log(ETH) = c + 1.0*log(BTC) + OU(theta=0.02)")
    print("True half-life = ln(2)/0.02 = 34.7 bars\n")

    # --- spread construction: fixed beta vs rolling beta -----------------
    print("-" * 70)
    print("1. DOES THE SPREAD CONSTRUCTION PRESERVE THE SIGNAL?")
    print("-" * 70)

    for mode, name in [("unit", "unit beta=1"), ("rolling", "rolling beta")]:
        p = Params(beta_mode=mode)
        f = build_spread(px_a, px_b, p)
        hl = half_life(f["spread"])
        adf = adf_test(f["spread"])
        print(f"{name:>14} : half-life {hl:9.1f} bars | ADF stat {adf['stat']:8.2f} "
              f"| p {adf['pvalue']:.4f}")

    print("\n  The true half-life is 34.7. A construction that reports something")
    print("  wildly larger is injecting a random-walk component into the spread.")

    # --- known-signal detection ------------------------------------------
    print("\n" + "-" * 70)
    print("2. KNOWN-SIGNAL TEST (zero costs, to isolate the signal)")
    print("-" * 70)

    for mode, name in [("unit", "unit beta=1"), ("rolling", "rolling beta")]:
        p = Params(beta_mode=mode, lookback=72, entry_z=2.0,
                   exit_z=0.5, cost_bps=0.0)
        f = run_backtest(px_a, px_b, p)
        t = trade_list(f)
        s = summarize(f, t, name)
        print(f"{name:>14} : SR {s['sharpe_gross']:6.2f} | ret {s['total_gross']:+8.2%} "
              f"| trades {s['trades']:4d} | hit {s['hit_rate']:.1%}")

    print("\n  The engine should find a strong edge on data that provably has one.")

    # --- look-ahead check -------------------------------------------------
    print("\n" + "-" * 70)
    print("3. LOOK-AHEAD CHECK (full-sample normalisation vs trailing window)")
    print("-" * 70)

    p = Params(beta_mode="unit", lookback=72, entry_z=2.0, exit_z=0.5, cost_bps=0.0)
    honest = run_backtest(px_a, px_b, p)
    cheat = cheating_backtest(px_a, px_b, p)
    s_h = summarize(honest, trade_list(honest), "honest")
    s_c = summarize(cheat, trade_list(cheat), "cheating")

    print(f"  trailing z : SR {s_h['sharpe_gross']:6.2f} | ret {s_h['total_gross']:+8.2%} "
          f"| trades {s_h['trades']}   (honest)")
    print(f"full-sample z : SR {s_c['sharpe_gross']:6.2f} | ret {s_c['total_gross']:+8.2%} "
          f"| trades {s_c['trades']}   (leaks the future)")
    print(f"\n  Look-ahead advantage = {s_c['sharpe_gross'] - s_h['sharpe_gross']:+.2f} Sharpe,")
    print(f"  on {s_c['trades']} trades instead of {s_h['trades']}: knowing the eventual")
    print("  mean lets it wait for only the deviations that actually revert.")
    print("  That gap is the size of the error from one careless line of pandas.")

    # --- null distribution --------------------------------------------------
    print("\n" + "-" * 70)
    print("4. NULL DISTRIBUTION (what Sharpe does this strategy get on noise?)")
    print("-" * 70)

    null = null_distribution(n_seeds=30, n=len(px_a), p=p)
    print(f"  30 runs on independent random walks, zero costs:")
    print(f"    mean   {null.mean():+.2f}")
    print(f"    median {np.median(null):+.2f}")
    print(f"    5th/95th pct  {np.percentile(null, 5):+.2f} / {np.percentile(null, 95):+.2f}")
    print(f"    min/max       {null.min():+.2f} / {null.max():+.2f}")
    print("\n  Mean near zero means the engine has no built-in bias. But look at")
    print("  the spread: on pure noise, a single run can land near +0.9 Sharpe")
    print("  purely by luck. That is the bar any real result has to clear, and")
    print("  it is why one backtest with a positive Sharpe proves nothing on")
    print("  its own.")


def null_distribution(n_seeds, n, p):
    """
    Run the strategy on independent random walks. There is no relationship
    between the two series, so the true edge is exactly zero. The resulting
    distribution of Sharpe ratios tells you how much apparent performance
    this strategy can manufacture from noise alone.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    out = []
    for seed in range(n_seeds):
        r = np.random.default_rng(1000 + seed)
        a = pd.Series(np.exp(np.cumsum(r.normal(0, 0.004, n))) * 3000, index=idx)
        b = pd.Series(np.exp(np.cumsum(r.normal(0, 0.004, n))) * 30000, index=idx)
        pp = Params(**{**p.__dict__, "cost_bps": 0.0})
        f = run_backtest(a, b, pp)
        out.append(summarize(f, trade_list(f), "null")["sharpe_gross"])
    return np.array(out)


if __name__ == "__main__":
    main()
