"""
Runs the full study:

  1. Load ETH and BTC hourly bars.
  2. Split chronologically into train (first 70%) and test (last 30%).
  3. Test the hypothesis statistically on TRAIN ONLY (ADF, half-life).
  4. Search parameters on TRAIN ONLY.
  5. Evaluate the single chosen parameter set on TEST, once.
  6. Report gross and net, with costs, plus a cost sensitivity sweep.

The discipline that matters: step 5 happens once. If you look at the test
result and then go back to step 4, the test set is contaminated and the
out-of-sample number is no longer out of sample.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from core.data import load_pair
from core.backtest import Params, run_backtest, trade_list, build_spread, estimate_beta
from core.metrics import summarize, adf_test, half_life, fmt_report, sharpe

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")


def split(idx, frac=0.7):
    cut = int(len(idx) * frac)
    return idx[:cut], idx[cut:]


def evaluate(px_a, px_b, p, label):
    f = run_backtest(px_a, px_b, p)
    t = trade_list(f)
    return f, t, summarize(f, t, label)


def search(px_a, px_b, base: Params, grid: dict, min_trades=15):
    """
    Grid search on the training set. Ranked by net Sharpe.

    min_trades guards against the classic overfit where a parameter set gets
    a beautiful Sharpe from three lucky trades.
    """
    keys = list(grid)
    results = []

    for combo in itertools.product(*(grid[k] for k in keys)):
        p = Params(**{**base.__dict__, **dict(zip(keys, combo))})
        f = run_backtest(px_a, px_b, p)
        t = trade_list(f)
        s = summarize(f, t, "train")
        s["params"] = dict(zip(keys, combo))
        results.append(s)

    df = pd.DataFrame(results)
    eligible = df[df["trades"] >= min_trades].copy()
    if eligible.empty:
        print(f"[search] No parameter set produced >= {min_trades} trades. Using all.")
        eligible = df.copy()

    eligible = eligible.sort_values("sharpe_net", ascending=False)
    return eligible


def cost_sweep(px_a, px_b, p: Params, bps_list):
    rows = []
    for bps in bps_list:
        pp = Params(**{**p.__dict__, "cost_bps": bps})
        f = run_backtest(px_a, px_b, pp)
        t = trade_list(f)
        s = summarize(f, t, f"{bps:.0f}bps")
        rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=720)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--sym-a", default="ETHUSDT")
    ap.add_argument("--sym-b", default="BTCUSDT")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--no-synthetic", action="store_true",
                    help="fail instead of falling back to synthetic data")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("ETH/BTC SPREAD MEAN REVERSION STUDY")
    print("=" * 72)

    a, b, source = load_pair(args.sym_a, args.sym_b, args.interval,
                             args.days, allow_synthetic=not args.no_synthetic)
    px_a, px_b = a["close"], b["close"]
    print(f"\nData source : {source}")
    print(f"Bars        : {len(px_a)}  ({px_a.index[0].date()} -> {px_a.index[-1].date()})")
    if source == "synthetic":
        print("\n*** SYNTHETIC DATA. These numbers describe a simulation, not the market. ***")
        print("*** Its purpose is to prove the pipeline finds a signal that is known  ***")
        print("*** to be there. Run with network access for the real result.          ***")

    train_idx, test_idx = split(px_a.index, args.train_frac)
    print(f"\nTrain       : {len(train_idx)} bars ({train_idx[0].date()} -> {train_idx[-1].date()})")
    print(f"Test        : {len(test_idx)} bars ({test_idx[0].date()} -> {test_idx[-1].date()})")

    # ---------------------------------------------------------------
    # Step 1: is the hypothesis even justified? Train data only.
    # ---------------------------------------------------------------
    # Hedge ratio is estimated ONCE, on train, and frozen. Re-estimating it
    # on test would leak test information into the model.
    beta_hat = estimate_beta(px_a.loc[train_idx], px_b.loc[train_idx])
    print(f"\nHedge ratio : beta = {beta_hat:.4f}  (OLS on train only, then frozen)")

    base = Params(beta_mode="fixed", fixed_beta=beta_hat)
    f_train_probe = build_spread(px_a.loc[train_idx], px_b.loc[train_idx], base)
    adf = adf_test(f_train_probe["spread"])
    hl = half_life(f_train_probe["spread"])

    print("\n" + "-" * 72)
    print("HYPOTHESIS CHECK (train only)")
    print("-" * 72)
    if adf.get("available"):
        verdict = "STATIONARY" if adf["stationary_at_5pct"] else "NOT stationary"
        print(f"ADF statistic     : {adf['stat']:.3f}   (5% critical {adf['crit_5pct']:.3f})")
        print(f"ADF p-value       : {adf['pvalue']:.4f}  ->  {verdict} at 5%")
        if not adf["stationary_at_5pct"]:
            print("  Note: the spread does not clearly mean-revert on this sample.")
            print("  Anything profitable below should be treated with suspicion.")
    else:
        print("ADF unavailable (statsmodels not installed).")
    print(f"OU half-life      : {hl:.1f} bars" if np.isfinite(hl) else "OU half-life: no reversion")

    # ---------------------------------------------------------------
    # Step 2: parameter search, train only.
    # ---------------------------------------------------------------
    grid = {
        "lookback": [72, 168, 336],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.0, 0.5, 1.0],
        "max_hold": [120, 240],
    }
    print("\n" + "-" * 72)
    print(f"PARAMETER SEARCH ON TRAIN  ({np.prod([len(v) for v in grid.values()])} combinations)")
    print("-" * 72)

    ranked = search(px_a.loc[train_idx], px_b.loc[train_idx], base, grid)
    top = ranked.head(5)
    for _, r in top.iterrows():
        print(f"  SR_net {r['sharpe_net']:6.2f} | net {r['total_net']:+7.2%} | "
              f"trades {int(r['trades']):3d} | {r['params']}")

    best_params_d = ranked.iloc[0]["params"]
    chosen = Params(**{**base.__dict__, **best_params_d})
    print(f"\nChosen: {best_params_d}")
    print(f"        cost_bps={chosen.cost_bps}, beta_mode={chosen.beta_mode}, "
          f"beta={chosen.fixed_beta:.4f}")

    # Spread of train Sharpes tells you how fragile the choice is.
    print(f"\nTrain SR_net across all {len(ranked)} eligible sets: "
          f"median {ranked['sharpe_net'].median():.2f}, "
          f"best {ranked['sharpe_net'].max():.2f}, "
          f"worst {ranked['sharpe_net'].min():.2f}")
    print("  A best that towers over the median is a warning sign of overfitting.")

    # ---------------------------------------------------------------
    # Step 3: evaluate ONCE on test.
    # ---------------------------------------------------------------
    f_tr, t_tr, s_tr = evaluate(px_a.loc[train_idx], px_b.loc[train_idx], chosen, "train")
    f_te, t_te, s_te = evaluate(px_a.loc[test_idx], px_b.loc[test_idx], chosen, "TEST (oos)")
    f_all, t_all, s_all = evaluate(px_a, px_b, chosen, "full")

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(fmt_report([s_tr, s_te, s_all]))

    print("\n" + "-" * 72)
    print("COST SENSITIVITY (test set)")
    print("-" * 72)
    sweep = cost_sweep(px_a.loc[test_idx], px_b.loc[test_idx], chosen, [0, 2, 5, 10, 20])
    print(fmt_report(sweep))

    # ---------------------------------------------------------------
    # How much of this could noise produce on its own?
    # ---------------------------------------------------------------
    from validate import null_distribution
    null = null_distribution(n_seeds=30, n=len(test_idx), p=chosen)
    pct = float((null >= s_te["sharpe_net"]).mean())
    print("\n" + "-" * 72)
    print("NULL CHECK (same strategy on 30 independent random walks)")
    print("-" * 72)
    print(f"Null Sharpe: median {np.median(null):+.2f}, "
          f"5th/95th pct {np.percentile(null,5):+.2f}/{np.percentile(null,95):+.2f}")
    print(f"Test Sharpe {s_te['sharpe_net']:+.2f} is beaten by {pct:.0%} of pure-noise runs.")
    if pct > 0.10:
        print("  This result is inside the range noise produces by luck.")

    # ---------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("READ-OUT")
    print("=" * 72)
    deg = s_tr["sharpe_net"] - s_te["sharpe_net"]
    print(f"Train net Sharpe : {s_tr['sharpe_net']:.2f}")
    print(f"Test  net Sharpe : {s_te['sharpe_net']:.2f}   (degradation {deg:+.2f})")
    print(f"Test  net return : {s_te['total_net']:+.2%} over {s_te['bars']} bars, "
          f"{s_te['trades']} trades")
    print(f"Costs paid       : {s_te['total_cost']:.2%} of notional turnover")

    if s_te["sharpe_net"] < 0.5:
        print("\nVerdict: does not hold up out of sample at realistic costs.")
        print("This is the expected outcome for a simple, widely known signal.")
    elif deg > 1.0:
        print("\nVerdict: large train-to-test degradation. Likely overfit to the train window.")
    else:
        print("\nVerdict: survives out of sample here. Next step would be to check")
        print("stability across other pairs and time periods before believing it.")

    # ---------------------------------------------------------------
    # Persist
    # ---------------------------------------------------------------
    f_all.to_csv(os.path.join(OUT_DIR, "bars.csv"))
    if len(t_all):
        t_all.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    ranked.drop(columns=["params"]).assign(
        params=ranked["params"].astype(str)
    ).to_csv(os.path.join(OUT_DIR, "param_search_train.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump({
            "source": source,
            "chosen_params": chosen.__dict__,
            "adf_train": adf,
            "half_life_train_bars": hl if np.isfinite(hl) else None,
            "train": s_tr, "test": s_te, "full": s_all,
            "cost_sweep_test": sweep,
        }, fh, indent=2, default=str)

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
