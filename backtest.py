"""
ETH/BTC ratio mean reversion backtest.

Hypothesis
----------
ETH and BTC are economically linked. The log price spread between them,
hedge-ratio adjusted, is stationary. When it deviates far from its recent
mean it tends to revert, so fading large deviations should be profitable
before costs.

Design notes on avoiding look-ahead bias
----------------------------------------
Every statistic used to make a decision at bar t is computed from data
strictly up to and including bar t, and the resulting position is applied
to the return from t to t+1. Concretely:

  - rolling beta  : estimated on window ending at t
  - rolling mean  : window ending at t
  - rolling std   : window ending at t
  - z-score at t  : (spread_t - mean_t) / std_t
  - position at t : decided from z_t
  - pnl at t+1    : position_t * spread_return_{t -> t+1}

The final `position.shift(1)` is the line that enforces this. Remove it and
the backtest will look spectacular and be wrong.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    lookback: int = 168          # hours used for rolling mean/std (1 week)
    beta_lookback: int = 336     # hours used for rolling hedge ratio (2 weeks)
    entry_z: float = 2.0         # enter when |z| exceeds this
    exit_z: float = 0.5          # exit when |z| falls back inside this
    stop_z: float = 4.0          # bail out if the spread keeps diverging
    max_hold: int = 240          # hard time stop, in bars
    cost_bps: float = 5.0        # one-way cost per leg, in basis points
    beta_mode: str = "fixed"     # "fixed" | "rolling" | "unit"
    fixed_beta: float = None     # set by estimate_beta() on the train window


def estimate_beta(px_a: pd.Series, px_b: pd.Series) -> float:
    """
    OLS hedge ratio from log(A) on log(B), estimated once.

    This must be called on the TRAINING window only, and the resulting number
    carried into the test window unchanged. Re-estimating it on the test data
    would leak test information into the model.
    """
    la, lb = np.log(px_a), np.log(px_b)
    df = pd.concat([la, lb], axis=1).dropna()
    slope = np.polyfit(df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy(), 1)[0]
    return float(slope)


def build_spread(px_a: pd.Series, px_b: pd.Series, p: Params) -> pd.DataFrame:
    """
    Spread = log(A) - beta * log(B).

    beta_mode:
      "unit"    beta = 1, so the spread is just log(A/B), the plain ratio.
      "fixed"   beta estimated once on the train window (see estimate_beta).
      "rolling" beta re-estimated on a trailing window every bar.

    A note on "rolling": it looks like the more sophisticated choice, but a
    beta estimated on a short trailing window is noisy, and that noise enters
    the spread multiplied by log(B), which is a random walk. The result is a
    spread that inherits a random-walk component and stops mean-reverting.
    validate.py measures this directly. "fixed" is the default for that reason.
    """
    la, lb = np.log(px_a), np.log(px_b)

    if p.beta_mode == "rolling":
        cov = la.rolling(p.beta_lookback).cov(lb)
        var = lb.rolling(p.beta_lookback).var()
        beta = (cov / var).rename("beta")
    elif p.beta_mode == "fixed":
        if p.fixed_beta is None:
            raise ValueError(
                "beta_mode='fixed' requires fixed_beta. Call estimate_beta() "
                "on the training window and pass the result in."
            )
        beta = pd.Series(p.fixed_beta, index=la.index, name="beta")
    elif p.beta_mode == "unit":
        beta = pd.Series(1.0, index=la.index, name="beta")
    else:
        raise ValueError(f"unknown beta_mode: {p.beta_mode}")

    spread = (la - beta * lb).rename("spread")

    mean = spread.rolling(p.lookback).mean()
    std = spread.rolling(p.lookback).std()
    z = ((spread - mean) / std).rename("z")

    return pd.DataFrame({
        "log_a": la, "log_b": lb, "beta": beta,
        "spread": spread, "mean": mean, "std": std, "z": z,
    })


def generate_positions(z: pd.Series, p: Params) -> pd.Series:
    """
    Stateful entry/exit. Position is in units of the spread:
      +1 = long spread  (long A, short beta*B)   entered when z is very low
      -1 = short spread (short A, long beta*B)   entered when z is very high
       0 = flat

    Exits on mean reversion (|z| < exit_z), on a stop (|z| > stop_z), or on
    a time limit. Iterating bar by bar keeps the state machine explicit and
    makes the look-ahead property easy to verify: the decision at index i
    only ever reads z.iloc[i].
    """
    pos = np.zeros(len(z))
    state = 0
    bars_held = 0
    zv = z.to_numpy()

    for i in range(len(zv)):
        zi = zv[i]

        if np.isnan(zi):
            pos[i] = 0
            state, bars_held = 0, 0
            continue

        if state == 0:
            if zi <= -p.entry_z:
                state, bars_held = 1, 0
            elif zi >= p.entry_z:
                state, bars_held = -1, 0
        else:
            bars_held += 1
            hit_target = abs(zi) <= p.exit_z
            hit_stop = abs(zi) >= p.stop_z
            hit_time = bars_held >= p.max_hold
            if hit_target or hit_stop or hit_time:
                state, bars_held = 0, 0

        pos[i] = state

    return pd.Series(pos, index=z.index, name="position")


def run_backtest(px_a: pd.Series, px_b: pd.Series, p: Params) -> pd.DataFrame:
    """Returns a per-bar frame with positions, gross/net returns and equity."""
    f = build_spread(px_a, px_b, p)
    f["position"] = generate_positions(f["z"], p)

    # Per-bar log returns of each leg.
    ret_a = f["log_a"].diff()
    ret_b = f["log_b"].diff()

    # Return on one unit of the spread, using the beta known at decision time.
    spread_ret = ret_a - f["beta"].shift(1) * ret_b

    # THE line that prevents look-ahead: yesterday's position earns today's return.
    held = f["position"].shift(1).fillna(0)
    f["gross_ret"] = held * spread_ret

    # Costs charged on turnover. Changing position by 1 unit trades 1 notional
    # of A and beta notional of B, so both legs are charged.
    turnover = held.diff().abs().fillna(held.abs())
    notional_traded = turnover * (1.0 + f["beta"].shift(1).abs().fillna(1.0))
    f["cost"] = notional_traded * (p.cost_bps / 10_000.0)

    f["net_ret"] = f["gross_ret"] - f["cost"]
    f["equity_gross"] = f["gross_ret"].fillna(0).cumsum()
    f["equity_net"] = f["net_ret"].fillna(0).cumsum()
    f["held"] = held

    return f


def trade_list(f: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-bar frame into one row per round-trip trade."""
    held = f["held"]
    changes = held.diff().fillna(held)
    trades = []
    open_i = None

    for i in range(len(held)):
        if held.iloc[i] != 0 and (open_i is None):
            open_i = i
        elif held.iloc[i] == 0 and open_i is not None:
            seg = f.iloc[open_i:i]
            trades.append({
                "entry_time": f.index[open_i],
                "exit_time": f.index[i],
                "direction": int(seg["held"].iloc[0]),
                "bars": len(seg),
                "entry_z": seg["z"].iloc[0],
                "exit_z": f["z"].iloc[i],
                "gross": seg["gross_ret"].sum(),
                "net": seg["net_ret"].sum(),
            })
            open_i = None

    if open_i is not None:  # still open at the end of the sample
        seg = f.iloc[open_i:]
        trades.append({
            "entry_time": f.index[open_i],
            "exit_time": f.index[-1],
            "direction": int(seg["held"].iloc[0]),
            "bars": len(seg),
            "entry_z": seg["z"].iloc[0],
            "exit_z": seg["z"].iloc[-1],
            "gross": seg["gross_ret"].sum(),
            "net": seg["net_ret"].sum(),
        })

    return pd.DataFrame(trades)
