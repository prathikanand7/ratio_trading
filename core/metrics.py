"""Performance metrics and the stationarity test that justifies the hypothesis."""

import numpy as np
import pandas as pd

BARS_PER_YEAR = 24 * 365  # hourly bars


def sharpe(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(bars_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Equity is a cumulative log-return curve, so drawdown is in log terms."""
    e = equity.dropna()
    if e.empty:
        return float("nan")
    return float((e - e.cummax()).min())


def summarize(f: pd.DataFrame, trades: pd.DataFrame, label: str) -> dict:
    net = f["net_ret"].dropna()
    gross = f["gross_ret"].dropna()
    exposure = (f["held"] != 0).mean()

    out = {
        "label": label,
        "bars": len(f),
        "start": f.index[0].date().isoformat() if len(f) else None,
        "end": f.index[-1].date().isoformat() if len(f) else None,
        "trades": len(trades),
        "exposure": exposure,
        "total_gross": float(gross.sum()),
        "total_net": float(net.sum()),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "max_dd_net": max_drawdown(f["equity_net"]),
        "total_cost": float(f["cost"].sum()),
    }

    if len(trades):
        out["hit_rate"] = float((trades["net"] > 0).mean())
        out["avg_trade_net"] = float(trades["net"].mean())
        out["avg_bars_held"] = float(trades["bars"].mean())
        out["best_trade"] = float(trades["net"].max())
        out["worst_trade"] = float(trades["net"].min())
    else:
        out.update({
            "hit_rate": float("nan"), "avg_trade_net": float("nan"),
            "avg_bars_held": float("nan"), "best_trade": float("nan"),
            "worst_trade": float("nan"),
        })

    return out


def adf_test(spread: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test on the spread.

    This is the check that separates "these two things are correlated" from
    "the gap between them actually reverts". A low p-value means we can
    reject a unit root, which is the statistical justification for trading
    mean reversion at all. A high p-value means the spread wanders and the
    whole hypothesis is on shaky ground, which is worth reporting honestly.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"available": False}

    s = spread.dropna()
    if len(s) < 50:
        return {"available": False}

    stat, pvalue, used_lag, nobs, crit, _ = adfuller(s, autolag="AIC")
    return {
        "available": True,
        "stat": float(stat),
        "pvalue": float(pvalue),
        "crit_5pct": float(crit["5%"]),
        "stationary_at_5pct": bool(stat < crit["5%"]),
        "nobs": int(nobs),
    }


def half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life of mean reversion, in bars.

    Regress delta_spread on lagged spread. If the slope is negative, the
    series pulls back toward its mean and the half-life tells you roughly
    how long a round trip takes, which is a sanity check on max_hold.
    """
    s = spread.dropna()
    if len(s) < 50:
        return float("nan")
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag, delta = lag.align(delta, join="inner")
    slope = np.polyfit(lag.to_numpy(), delta.to_numpy(), 1)[0]
    if slope >= 0:
        return float("inf")
    return float(-np.log(2) / slope)


def fmt_report(rows: list) -> str:
    """Render a list of summarize() dicts as a readable table."""
    keys = [
        ("label", "period", "{}"),
        ("bars", "bars", "{}"),
        ("trades", "trades", "{}"),
        ("exposure", "exposure", "{:.1%}"),
        ("total_gross", "gross ret", "{:+.2%}"),
        ("total_net", "net ret", "{:+.2%}"),
        ("sharpe_gross", "SR gross", "{:.2f}"),
        ("sharpe_net", "SR net", "{:.2f}"),
        ("max_dd_net", "max DD", "{:.2%}"),
        ("hit_rate", "hit rate", "{:.1%}"),
        ("avg_bars_held", "avg hold", "{:.0f}h"),
    ]
    header = " | ".join(f"{h:>10}" for _, h, _ in keys)
    lines = [header, "-" * len(header)]
    for r in rows:
        cells = []
        for k, _, fmt in keys:
            v = r.get(k)
            try:
                cells.append(f"{fmt.format(v):>10}")
            except (ValueError, TypeError):
                cells.append(f"{'n/a':>10}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)
