# Research Report: ETH/BTC Spread Mean Reversion

**Outcome: REJECTED** (ETH/BTC). Exploratory multi-pair scan: **INCONCLUSIVE**.
**Researcher:** Prathik Anand  |  **Report date:** 2026-10-15  |  **SOP version:** 1.1

## 1. Research definition (SOP s.1)

| Item | Definition |
|---|---|
| Hypothesis | The hedge-ratio-adjusted log spread between ETH and BTC is stationary, so large deviations from a rolling mean revert and can be faded. |
| Target | Spread return (long/short one unit of ETH against beta units of BTC). |
| Holding period | Hours to days (hourly bars, average hold about 120h, time stop 240h). |
| Universe | ETHUSDT and BTCUSDT spot (Binance). Exploratory scan: 10 large-cap USDT pairs. |
| Economic rationale | Both assets share market risk, so their relative price may be anchored. Weakness: no fundamental anchor between the two, so the link can drift. |
| Benchmark | **Not agreed.** SOP specifies SPY for US only; crypto benchmark to be set by Kairosys (open question). |
| Strategy type | Market-neutral long/short (pairs). |

## 2. Data and point-in-time integrity (s.2)

- **Source:** Binance Vision public monthly kline archives, 1h bars, 2024-10-01 to 2026-09-30 (17,520 bars per symbol, no gaps). This is **not** the Kairosys raw store; approval of the source is an open question for Kary.
- **Timestamps:** close time, UTC. Binance switched from ms to microseconds partway through the archive; unit is detected by magnitude and covered by unit tests (`test_data.py`, all passing).
- **Alignment:** inner join on timestamps; nothing was dropped.
- **Look-ahead:** the signal at bar t uses data up to t; position is shifted one bar before earning returns. The hedge ratio is estimated on the training window only and then frozen.
- **Survivorship:** the scan universe was chosen in 2026 from coins that exist today, so it is survivorship-biased. Not corrected.
- **Revisions:** spot kline archives are not restated; no corporate actions apply.

## 3. Method (s.3, s.4)

- Features: one price-based feature, the rolling z-score of the spread (lookback 72/168/336 bars).
- Baseline: the single rule-based strategy itself; no ML.
- Split: chronological 70/30 train/test (train to 2026-02-23, test 2026-02-24 to 2026-09-30).
- Walk-forward: 6-month train, 2-month test, rolled by 2 months, giving 9 out-of-sample windows (2025-03-30 to 2026-09-30, about 18 months). Parameters and hedge ratio chosen on train only in each window.
- Costs: 5 bps per leg per trade, swept 0 to 20 bps. **No slippage, borrow or funding costs modelled.**
- Code: `data.py`, `backtest.py`, `metrics.py`, `run_study.py`, `walkforward.py`, `scan_pairs.py`.

## 4. Experiments disclosed (s.4, s.7)

| Experiment | Count |
|---|---|
| Primary hypothesis (ETH/BTC) | 1 |
| Parameter grid (lookback x entry_z x exit_z x max_hold) | 54 configurations |
| Walk-forward fits for ETH/BTC | 9 windows x 54 = 486 |
| Exploratory pairs tested for cointegration | 45 |
| Walk-forward on the 3 lowest-p pairs | 3 x 9 x 54 = 1,458 fits |
| Cost sensitivity settings | 5 |
| Noise-comparison runs (random walks) | 30 |

The 54-set grid, the 45-pair scan and the 3 pairs carried to walk-forward are all material degrees of freedom. Any positive result from them must be read as best-of-many.

## 5. Results

### 5.1 Hypothesis check (train only)

| Test | Result |
|---|---|
| ADF on spread | statistic -1.61 vs 5% critical -2.86, p = 0.477: **not stationary** |
| OU half-life | about 2,175 bars (~90 days), versus ~120h average hold |
| Hedge ratio (OLS, frozen) | 1.036 |

### 5.2 Single split

Returns below are sums of log returns on one unit of notional; drawdown is in log terms (simple drawdown is smaller, e.g. full-period -0.69 log is about -50%).

| Period | Net return (log) | Net Sharpe | Max DD (log) | Trades |
|---|---|---|---|---|
| Train | +13.0% | 0.24 | -0.69 | 77 |
| Test (OOS) | +15.5% | 1.22 | -0.13 | 32 |
| Full | +34.9% | 0.50 | -0.69 | 111 |

Noise check: 33% of 30 random-walk runs beat the test Sharpe of 1.22, so this result is within what chance produces. Cost sensitivity (test): net return +21.9% at 0 bps, +15.5% at 5, +9.1% at 10, **-3.7% at 20 bps**.

### 5.3 Walk-forward (ETH/BTC, 9 windows)

- Profitable windows: **4 of 9**. Median window Sharpe: **-0.03**.
- Stitched OOS net return: +10.6% (log) over about 18 months; annualised Sharpe **0.23**.
- Train Sharpe (1.0 to 3.1) did not predict test Sharpe (-3.3 to +4.3). One window (+26%, 9 trades) provides most of the gain.
- Chosen parameters and hedge ratio (0.73 to 2.75) change nearly every window, indicating instability.

### 5.4 Exploratory pair scan (not part of the primary hypothesis)

Engle-Granger cointegration on the train window for 45 pairs; Bonferroni threshold p < 0.0011. **No pair passed** (lowest: AVAX/DOGE p = 0.030, LTC/ADA p = 0.031; about two pairs below 0.05 are expected by chance).

Walk-forward on the three lowest-p pairs, for reference only:

| Pair | Profitable windows | Median SR | Stitched OOS (log) | Annualised SR |
|---|---|---|---|---|
| AVAX/DOGE | 7/9 | 2.96 | +68% | 1.46 |
| LTC/ADA | 6/9 | 0.90 | +21% | 0.43 |
| LINK/LTC | 5/9 | 1.31 | +36% | 0.61 |

These are best-of-45 selections and the cointegration window overlaps the walk-forward windows, so they are not evidence of an edge.

## 6. Acceptance criteria (s.6), ETH/BTC

| Criterion | Requirement | Result | Status |
|---|---|---|---|
| OOS length / regimes | >= 5 years, >= 2 regimes | ~18 months walk-forward, 7 months single split; 2 years of data | Fail |
| Sharpe | >= 1.0 | 0.23 (walk-forward) | Fail |
| Max drawdown | <= 30% | about -50% simple (full period) | Fail |
| CAGR | >= 10% | about 7% (walk-forward, log, unit notional) | Fail |
| Alpha vs benchmark, t-stat | >= 2% / >= 2.0 | Not computed (no benchmark agreed) | Not assessed |
| Calmar, Sortino, 95% CIs | >= 0.5 / >= 1.2 / report | Not computed | Not assessed |

Sharpe uses a zero risk-free rate (not subtracted); acceptable only as an approximation for a market-neutral book.

## 7. Known limitations

- Two years of data cannot cover the s.6 requirement of five years and two regimes.
- The original single-split OOS period was inspected and then reused in the walk-forward, so it is no longer an untouched holdout (s.8).
- Returns are per unit of notional, not on posted capital; gross notional is about 2x per unit of spread.
- No slippage, short-leg funding or borrow costs; spot-only; costs of 5 bps are an assumption (crypto is not in the s.5 cost table).
- Survivorship bias in the scan universe; multiple-testing correction applied only to the cointegration step.

## 8. Conclusion and next steps

**ETH/BTC: REJECTED.** The spread is not stationary (ADF p = 0.48), the walk-forward Sharpe is 0.23 with 4/9 profitable windows, drawdown breaches the limit, and the one good out-of-sample split is within the range pure noise produces.

**Pair scan: INCONCLUSIVE.** No pair is cointegrated after correction. AVAX/DOGE is the only candidate worth a follow-up, with parameters frozen and tested on data published after 2026-09-30 (October 2026 onward), which should be reserved as the untouched holdout.

**To confirm with Kairosys (Kary):** (1) whether crypto is in scope; (2) whether Binance Vision data is acceptable or should come from the raw store; (3) the benchmark and transaction-cost assumption for crypto.
