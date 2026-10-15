# Research Report 2: Economically Linked Pairs and Spot/Perp Carry

**Outcomes**
- Economically linked pairs (mean reversion): **REJECTED** (UNI/AAVE noted as borderline, not supported).
- Spot/perp cash-and-carry: **PROMISING (low-yield carry), not PASS.** Positive and stable after costs, but about 5% a year on the holdout, below the SOP return bar, and not alpha.

**Researcher:** Prathik Anand | **Report date:** 2026-10-15 | **SOP version:** 1.1
**Follows:** `SOP_Research_Report.md` (ETH/BTC rejected, 45-pair scan inconclusive). Experiment counts below are in addition to that report.

## 1. Research definition (SOP s.1)

| Item | Study A: linked pairs | Study B: cash-and-carry |
|---|---|---|
| Hypothesis | Pairs with an economic link have a stationary log spread that can be faded. | A long-spot / short-perp book earns the perp funding rate with little price risk, because funding ties the perp to spot. |
| Target | Spread return of a hedge-ratio-weighted long/short pair. | Net hourly return on 1 unit of spot notional: spot return minus perp return, plus funding, minus costs. |
| Holding period | Hours to days (average hold about 5 days). | Weeks to months (always-on), or days to weeks (filtered). |
| Universe | LTC/BCH, XLM/XRP, SOL/AVAX, UNI/AAVE, ETC/ETH, plus ETH/BTC as a reference. Binance USDT spot. | BTC, ETH, SOL, BNB, XRP, DOGE. Binance spot against USDT-M perpetual. |
| Economic rationale | Shared sector, narrative or codebase. Weak: no contractual anchor, so the link can drift. | Perpetual funding is paid by longs to shorts when the perp trades rich, so the short collects it. Strong anchor, but the return pays for supplying leverage to longs and shrinks when leverage demand falls. |
| Benchmark | **Not agreed.** Crypto benchmark to be set by Kairosys. | **Not agreed.** A cash or risk-free-rate comparison is the natural one; see s.6. |
| Strategy type | Market-neutral long/short. | Market-neutral carry (delta-neutral). |

The pair list for Study A and the two rules for Study B were written down before results were seen. There was no parameter search in Study B.

## 2. Data and point-in-time integrity (s.2)

- **Source:** Binance Vision public archives. Spot 1h klines, USDT-M perpetual 1h klines (`futures/um`), and perpetual funding-rate settlements. Window 2020-10-01 to 2026-09-30 (72 months). This is **not** the Kairosys raw store; approval of the source is an open question.
- **Split:** DEV is everything before 2024-10-01 (Oct 2020 to Sep 2024). HOLDOUT is 2024-10-01 onward. All design and first results used DEV only. The holdout was evaluated once, with rules unchanged.
- **Holdout caveat:** the 2024-10 onward window was already used for the ETH/BTC study and the 10-coin scan in Report 1. It was not used for any of the pairs or the carry strategy here, but it is not a pristine holdout for the coins involved.
- **Timestamps:** bar END time, UTC (close time rounded to the hour). Funding settlements (00:00, 08:00, 16:00 UTC) are matched to the bar ending at that time. Spot and perp series are inner-joined, with duplicates dropped.
- **Gaps:** about 11 to 12 irregular timestamps per pair in DEV. They were not filled. A return across a gap is treated as one return.
- **Look-ahead:** signals use data up to bar t and the position starts on bar t+1. In Study A the hedge ratio and parameters are fitted on the trailing 6-month train window only. In Study B the filtered rule uses funding settlements up to the decision bar.
- **Survivorship:** the assets are large coins that exist today and were chosen in 2026. Not corrected.

## 3. Method (s.3, s.4)

**Study A.** The same walk-forward design as Report 1. Train on 6 months, trade the next 2 months, roll by 2 months, with parameters and hedge ratio chosen on train only (grid of 54: lookback 72/168/336, entry z 1.5/2.0/2.5, exit z 0/0.5/1.0, max hold 120/240). This gives 21 out-of-sample windows per pair on DEV. A cointegration check (Engle-Granger) is reported for information. The threshold is Bonferroni-adjusted to 0.05/5 = 0.010.

**Study B.** Fixed rules, no fitted parameters:
- *always_on:* hold the hedge for the whole period.
- *filtered:* hold only while the trailing 7-day mean funding (last 21 settlements) is positive.

P&L per hour on 1 unit of spot notional is `spot return - perp return + funding at settlement - costs`.

**Costs (assumed, not from the SOP tables):** spot 10 bps and perp 4.5 bps taker fees per trade (base tier, so conservative for a larger account only if VIP tiers are not available), plus 2 bps slippage. Opening or closing the hedge trades both legs. No borrow cost applies, because the long leg is spot.

**Code:** `data.py`, `backtest.py`, `metrics.py`, `walkforward.py`, `scan_linked.py` (Study A), `basis.py` (Study B).

## 4. Experiments disclosed (s.4, s.7)

| Experiment | Count |
|---|---|
| Study A: pre-registered pairs (plus ETH/BTC reference) | 5 (+1) |
| Study A: walk-forward fits | 6 pairs x 21 windows x 54 parameter sets = 6,804 |
| Study B: assets x rules, evaluated on DEV | 6 x 2 = 12 |
| Study B: same 12, evaluated once on HOLDOUT | 12 |
| Debugging reruns of Study B on DEV (data-alignment fix) | 2 (no rule or parameter changed) |

Cumulative with Report 1 (45-pair scan, 54-set grids, 30 noise runs), the experimentation level is high for Study A. A positive Study A result would be best-of-many and would need a fresh holdout. Study B has few degrees of freedom.

## 5. Results

### 5.1 Study A: linked pairs (DEV, walk-forward, 21 windows)

| Pair | Coint. p (full dev) | Profitable windows | Median window SR | Annualised SR |
|---|---|---|---|---|
| UNI/AAVE | 0.055 | 14/21 | 0.82 | 0.99 |
| LTC/BCH | 0.404 | 9/21 | -0.19 | 0.11 |
| XLM/XRP | 0.768 | 9/21 | -0.27 | -0.38 |
| SOL/AVAX | 0.673 | 7/21 | -0.71 | -0.79 |
| ETC/ETH | 0.421 | 9/21 | -0.83 | -1.10 |
| ETH/BTC (reference) | 0.437 | 11/21 | 0.23 | -0.03 |

- **No pair is cointegrated** at the corrected threshold (0.010). The lowest is UNI/AAVE at 0.055.
- **Four of five pairs lose money** in walk-forward. Stitched out-of-sample returns are in the -68% to -271% range for XLM/XRP, SOL/AVAX and ETC/ETH (log, unit notional).
- **UNI/AAVE** is the only positive result. Its cointegration is borderline, its yearly returns are uneven (2023: +0.4%), and it is one of five tested pairs. Treated as not supported. It was **not** carried to the holdout, because nothing here meets the pre-set bar for it.

### 5.2 Study B: carry, DEV (2020-10 to 2024-09)

Returns are annualised mean net return on spot notional. Sharpe uses a zero risk-free rate.

| Asset | Mean funding / 8h | always_on ann. | SR | Max DD | filtered ann. | SR | Max DD |
|---|---|---|---|---|---|---|---|
| BTC | 1.28 bps | +13.9% | 4.52 | -2.0% | +12.4% | 4.00 | -2.0% |
| ETH | 1.45 bps | +15.6% | 4.12 | -2.5% | +14.6% | 3.95 | -2.5% |
| XRP | 1.61 bps | +16.6% | 2.26 | -8.2% | +15.7% | 2.72 | -5.3% |
| DOGE | 1.52 bps | +17.3% | 2.09 | -4.4% | +14.9% | 2.02 | -4.8% |
| SOL | -0.03 bps | -3.2% | -0.16 | -52.0% | +10.9% | 1.04 | -15.9% |
| BNB | -0.03 bps | -0.5% | -0.08 | -27.0% | +8.0% | 1.64 | -5.4% |

SOL's always-on loss in 2022 (-45%) was checked: funding summed to about -38% that year (SOL perp traded at a persistent discount, so shorts paid), plus about -7% from the spot/perp hedge during the Nov 2022 dislocation. It is a real feature of the data, not a coding error. The filter avoided most of it on DEV.

### 5.3 Study B: carry, HOLDOUT (2024-10 to 2026-09), evaluated once

| Asset | Mean funding / 8h | always_on ann. | SR | Max DD | filtered ann. | SR | Max DD |
|---|---|---|---|---|---|---|---|
| BTC | 0.48 bps | +5.2% | 4.71 | -0.4% | +3.5% | 3.02 | -2.3% |
| ETH | 0.45 bps | +4.8% | 4.01 | -0.7% | +2.9% | 2.22 | -2.4% |
| DOGE | 0.46 bps | +5.0% | 2.77 | -0.6% | +2.5% | 1.35 | -2.3% |
| XRP | 0.31 bps | +3.4% | 2.02 | -1.3% | -0.3% | -0.17 | -6.3% |
| SOL | 0.11 bps | +1.1% | 0.54 | -4.0% | -3.1% | -1.54 | -8.8% |
| BNB | 0.09 bps | +0.9% | 0.47 | -3.0% | +1.5% | 1.15 | -1.4% |

**Reading the DEV-to-holdout change**
- On BTC, ETH, DOGE and XRP, always-on stays positive with Sharpe 2 to 4.7 and drawdown under 1.5%. The level of return fell by about two thirds, from 14 to 17% to 3 to 5%, because mean funding fell to roughly 0.3 to 0.5 bps per 8h from 1.3 to 1.6 bps.
- The filtered rule **failed out of sample**. It was worse than always-on on five of six assets (SOL, XRP, BTC, ETH, DOGE), because switching costs outweigh the funding avoided when funding is small. Its DEV advantage on SOL and BNB did not hold. The filter is rejected.
- SOL and BNB carry about 1% a year always-on, which is not worth the operational cost.

## 6. Acceptance criteria (s.6)

Assessed for BTC/ETH always-on carry, the best candidate.

| Criterion | Requirement | Result | Status |
|---|---|---|---|
| OOS length / regimes | >= 5 years, >= 2 regimes | DEV 4 years (2021 bull, 2022 bear, 2023 chop) plus 2-year holdout, 6 years in total, but only the holdout is out of sample | Partial |
| Sharpe | >= 1.0 | 4.0 to 4.7 (holdout, zero risk-free rate) | Pass |
| Max drawdown | <= 30% | under 1% (holdout); about 2 to 2.5% (DEV) | Pass (see s.7: margin risk not modelled) |
| CAGR | >= 10% | about 5% (holdout); 14 to 16% (DEV) | **Fail on holdout** |
| Alpha vs benchmark, t-stat | >= 2% / >= 2.0 | Not computed (no benchmark agreed) | Not assessed |
| Calmar, Sortino, 95% CIs | >= 0.5 / >= 1.2 / report | Not computed | Not assessed |

**Risk-free caveat.** The Sharpe ratios use a zero risk-free rate. If USD cash yields about 4% (an assumption, not checked against data), the holdout excess return over cash is close to zero, and the Sharpe would fall sharply. This is the largest open point for a benchmark decision.

Report-only items: turnover is near zero for always-on (no switches) and 17 to 77 switches over the holdout for the filtered rule. Beta to the underlying is about zero by construction (delta-neutral).

## 7. Robustness and known limitations (s.7, s.10)

- **Regime dependence.** Carry earned 14 to 17% in a high-leverage market (2021) and 3 to 5% in the recent low-funding market. It can turn negative (SOL 2022; BNB with 23% positive funding). The holdout is one regime, a low-funding one.
- **Margin and liquidation risk not modelled.** The short perp needs posted margin and can be liquidated in a spike even if the book is hedged. Drawdowns shown are mark-to-market on the hedge only.
- **Returns are on spot notional,** not on posted capital. Real capital includes perp margin, so capital returns are roughly 15 to 20% lower.
- **Simplifications.** Hourly rebalancing of the hedge to constant notional is assumed to be free. Funding is assumed to be received on the full notional at each settlement. Slippage is a flat assumption.
- **Counterparty and operational risk.** Single exchange (Binance). Withdrawal, outage and exchange-failure risk are not quantified.
- **Costs.** Fees are base-tier assumptions. Crypto is not in the s.5 cost table.
- **Survivorship and data.** Assets are today's large coins. Source is public Binance data, not the Kairosys raw store.
- **Study A.** Heavy experimentation (6,804 fits); the best result (UNI/AAVE) is borderline and uneven, and it was not validated on the holdout.
- **Sharpe** is computed on hourly returns annualised with sqrt(8,760), with a zero risk-free rate.

## 8. Conclusion and recommendation

- **Linked pairs: REJECTED.** None is cointegrated after correction, four of five lose money in walk-forward, and the reference ETH/BTC again shows no edge.
- **Cash-and-carry: PROMISING (low-yield carry).** It is positive in most years, has very low drawdown in the dev and holdout periods, and has a clear economic mechanism. It does not qualify as PASS: the holdout return (about 5%) is below the 10% CAGR minimum, benchmark alpha is not assessable, and it is carry rather than alpha.
- **Rejected inside the study:** the funding filter, and SOL and BNB as carry assets.

**Questions for Kairosys (Kary)**
1. Is crypto in scope, and is public Binance data acceptable, or should it come from the raw store?
2. What benchmark and transaction-cost assumption apply to crypto? In particular, is a cash or risk-free rate the right comparison for a market-neutral carry?
3. Is a delta-neutral carry of about 5% a year (before margin and exchange risk) of interest, given the SOP return bar of 10% CAGR? If so, the next steps are modelling margin and liquidation, a return-on-capital view, and a longer history from a source that includes pre-2020 data, if the SOP's 5-year out-of-sample requirement is to be met.

## 9. Reproducibility

```
python scan_linked.py           # Study A (DEV only)
python basis.py                 # Study B on DEV
python basis.py --holdout       # Study B on HOLDOUT (run once)
```

Data is cached under `cache/` (parquet). Python deps are in `requirements.txt` plus `statsmodels` and `pyarrow`.
