# ETH/BTC Spread Mean Reversion

A small, honest research study: form a hypothesis, test it on data, validate
out of sample, and report what actually happened.

Built as the research-side counterpart to a live `RatioTradingAlgorithm` that
computes an ETH/BTC ratio from order book mid prices and fires a trade when it
crosses a fixed threshold. That execution module takes `target_ratio` and
`threshold` as human-supplied inputs. This repo is the attempt to answer where
those numbers should come from.

## Hypothesis

ETH and BTC are economically linked. The hedge-ratio-adjusted log price spread
between them is stationary, so large deviations from its recent mean tend to
revert. Fading those deviations should be profitable before costs.

Two things have to be true for this to be worth trading, and they are tested
separately:

1. The spread is actually stationary (ADF test, train data only).
2. The reversion is large enough and fast enough to survive fees and slippage.

## Files

| File | What it does |
|---|---|
| `data.py` | Pulls hourly klines from Binance Vision (bulk CSV archive), with the REST API and synthetic data as fallbacks. |
| `test_data.py` | Tests for the Vision parser: header sniffing, timestamp units, bar spacing, inner join. |
| `backtest.py` | Spread construction, z-score signal, position state machine, PnL with costs. |
| `metrics.py` | Sharpe, max drawdown, hit rate, ADF stationarity test, OU half-life. |
| `run_study.py` | The full study: split, test hypothesis, search on train, evaluate once on test. |
| `validate.py` | Four checks on the engine itself. Run this first. |

## Running it

```bash
pip install pandas numpy statsmodels pyarrow

python test_data.py                # check the data parser
python validate.py                 # check the engine before trusting it
python run_study.py --days 720     # the actual study
```

## Where the data comes from

`data.py` tries three sources in order and names whichever one it used.

**1. Binance Vision** (`data.binance.vision`) is the primary source and the
one to use. It is a free static file host, no API key, no account, no rate
limit, holding every kline Binance has ever published as monthly zipped CSVs:

```
https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1h/ETHUSDT-1h-2025-03.zip
```

Two years of hourly ETH and BTC is 48 files and downloads in under a minute.
Results are cached to Parquet, so the second run is instant. The current month
is skipped, because Vision publishes a monthly archive only once the month has
ended.

**2. Binance REST** (`api.binance.com/api/v3/klines`) is the fallback. Also
keyless, but capped at 1000 bars per call, so multi-year history means paging
with a sleep between calls. Its real use is topping up the current month that
Vision has not published yet.

**3. Synthetic** is the last resort, so the pipeline can be validated with no
network at all. It prints a loud warning. Those numbers describe a simulation,
not the market.

Pick one explicitly with `load_pair(..., source="vision")` if you want to.

### Other sources worth knowing

- **ccxt** (`pip install ccxt`) wraps 100+ exchanges behind one interface. Good
  if you want to repeat this on Kraken or Coinbase to check the result is not
  a Binance artifact.
- **Kraken** publishes free quarterly OHLCVT dumps going back to 2013, which is
  further than most. Useful for testing whether a relationship held across
  earlier market regimes.
- **Tardis.dev** has full order book and tick data, including some free
  samples. Only worth it if you move to a horizon short enough that bar data
  hides what you need, which this strategy is not.
- **Deribit, OKX and Bybit** all publish free funding rate history, which is
  what the alternative hypothesis in the backlog needs.

### One thing to be careful about

Do not use your employer's proprietary market data, recorded feeds or internal
code for a personal project you intend to show to another firm. Everything
above is public and free, so there is no reason to go near it.

## Method, and the discipline behind it

**Chronological split.** First 70% is train, last 30% is test. No shuffling,
because shuffling time series data leaks the future into the past.

**Hedge ratio estimated once.** Beta comes from an OLS of `log(ETH)` on
`log(BTC)` over the training window, then is frozen. Re-estimating it on test
data would leak test information into the model.

**Parameters searched on train only.** The grid covers lookback, entry
threshold, exit threshold and max holding period. The winner is chosen by net
Sharpe, with a minimum trade count so a parameter set cannot win on three lucky
trades.

**Test evaluated once.** This is the part that is easy to write and hard to do.
Looking at the test result and then adjusting parameters turns the test set
into a second training set, and the out-of-sample number stops meaning
anything.

**Costs included from the start.** Default 5 bps one way per leg, with a
sensitivity sweep from 0 to 20 bps. A strategy that only works at zero cost is
not a strategy.

## Avoiding look-ahead bias

Every statistic used to decide a position at bar `t` is computed from data up
to and including `t`, and that position earns the return from `t` to `t+1`.
The line that enforces it:

```python
held = f["position"].shift(1)      # yesterday's decision earns today's return
f["gross_ret"] = held * spread_ret
```

`validate.py` tests this rather than asserting it.

## What validate.py checks

**1. Spread construction preserves the signal.** On synthetic data built with a
known OU residual of half-life 34.7 bars, a unit-beta spread recovers 32.7. A
*rolling* beta reports 5691. That result killed the original design: a hedge
ratio estimated on a short trailing window is noisy, and the noise enters the
spread multiplied by `log(BTC)`, which is a random walk. The spread inherits a
random-walk component and stops reverting. Backtest Sharpe on the same data
fell from 4.84 with unit beta to 0.55 with rolling beta. Hence beta is
estimated once on train and frozen.

**2. Known-signal detection.** On data that provably contains an edge, the
engine finds it. An engine that cannot find a signal that is definitely there
cannot be trusted to say a signal is absent.

**3. Look-ahead.** Compares the trailing z-score against a full-sample one
(`(spread - spread.mean()) / spread.std()`, which quietly tells every bar where
the spread will average out over the whole period). The cheating version scores
+2.23 Sharpe higher on a third of the trades: knowing the eventual mean lets it
wait for only the deviations that actually revert.

Worth noting what is *not* a good test of this: simply dropping the `.shift(1)`
makes results look *worse*, not better, because it pays each position the move
that triggered its own entry. For a contrarian strategy that is a loss. "My
results got worse when I broke it" is not proof a backtest is clean.

**4. Null distribution.** The same strategy on 30 independent random walks,
where the true edge is exactly zero. Mean Sharpe comes out at +0.02, so the
engine has no built-in bias. But the 5th-95th percentile range is roughly -0.94
to +0.83. On pure noise, a single run can reach +0.9 Sharpe by luck. That is
the bar any real result has to clear, and it is why one backtest with a
positive Sharpe proves nothing.

`run_study.py` runs this null check against the actual test result and reports
what fraction of noise runs beat it.

## Reading the output

On synthetic data with a deliberately planted signal, the pipeline reports
train Sharpe 6.73, test 6.65, degradation 0.08, beaten by 0% of noise runs.
That is what a correct engine looks like on data that genuinely has an edge.

On real ETH/BTC data, expect something far weaker. Simple pairs trading on the
most liquid crypto pair is thoroughly mined, and the most likely honest finding
is that the signal exists gross but does not survive costs. Signs a result
should not be believed:

- Test Sharpe far below train Sharpe (overfit to the train window)
- Test Sharpe inside the null distribution's range (indistinguishable from luck)
- Profitable at 0 bps, unprofitable at 5 (the edge is smaller than the spread)
- Very few trades carrying most of the PnL
- ADF failing to reject a unit root, which removes the justification for the
  whole approach

## Known limitations

- **Fills are assumed at the bar close.** No slippage model beyond a flat bps
  charge, no order book simulation, no partial fills. The live execution module
  this study was built alongside handles that; the backtest does not.
- **No funding costs.** Holding a short leg on perps has a funding cost that is
  not modelled. On a strategy holding ~30 hours per trade this matters.
- **Constant position size.** No volatility targeting, no position scaling with
  signal strength.
- **One pair, one period.** Even a result that survives here would need to hold
  across other pairs and other time windows before meaning much.
- **Beta is static across the whole test window.** A real relationship drifts.
  The fix is periodic re-estimation on a schedule known in advance, not a
  trailing window, for the reason in check 1 above.

## Next steps

1. Run on real data and see whether the ADF test even justifies the hypothesis.
2. If the spread is stationary, check whether reversion survives costs.
3. If it does, test stability across pairs (SOL/ETH, BNB/BTC) and periods.
4. Only then think about what `target_ratio` and `threshold` should be in the
   live execution module.
