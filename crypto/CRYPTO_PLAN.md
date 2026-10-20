# AV Crypto Pull — Plan

**Scope:** DIGITAL_CURRENCY_DAILY for 6 assets × 8 fiat markets.  
**Data source:** Alpha Vantage **Premium 75** plan (same key as SPY options and commodities).  
**Status:** Scripts built and wired to SessionStart. Pull ready.  
**Author note:** API key lives in `~/.kairosys.env` as `ALPHAVANTAGE_API_KEY`, read via `config.py`. Never put it in this file or any committed file.

---

## 1. Scope

| Asset | AV supported | Data from | Max depth | Notes |
|---|---|---|---|---|
| BTC | ✓ | 2010-07-17 | 16 years | |
| ETH | ✓ | 2015-08-08 | 11 years | |
| SOL | ✓ | 2020-04-09 | 6 years | Launched 2020 — can't get more |
| XRP | ✓ | 2016-01-01 | 10 years | |
| DOGE | ✓ | 2016-01-01 | 10 years | |
| ZEC | ✓ | 2016-10-30 | 10 years | |
| BNB | ✗ | — | — | AV returns invalid API error |
| TRX | ✗ | — | — | AV returns invalid API error |

**Markets attempted:** USD, EUR, GBP, CNY, JPY, CHF, CAD, AUD (8 per asset)  
**Markets confirmed supported by AV (from pull 2026-10-19):**

| Asset | Supported markets | Unsupported (api_error: av_unsupported_pair) |
|---|---|---|
| BTC | USD, EUR, GBP | CNY, JPY, CHF, CAD, AUD |
| ETH | USD, EUR, GBP | CNY, JPY, CHF, CAD, AUD |
| SOL | USD, EUR, GBP | CNY, JPY, CHF, CAD, AUD |
| XRP | USD, EUR, GBP | CNY, JPY, CHF, CAD, AUD |
| DOGE | USD, EUR, GBP | CNY, JPY, CHF, CAD, AUD |
| ZEC | USD only | EUR, GBP, CNY, JPY, CHF, CAD, AUD |

16 of 48 attempted pairs return data. The script still attempts all 48 on each run so any new market support from AV is picked up automatically.

**Excluded:**
- `CRYPTO_INTRADAY`: the `month=` parameter is silently ignored by AV — always returns the same ~350 bars (~1 day) regardless. No historical intraday available.
- `CURRENCY_EXCHANGE_RATE`: real-time snapshot only, no time series history.

---

## 2. API tier and call budget

**Premium 75 plan.** Each call returns full history in one response — no date loop.

| | Count | Notes |
|---|---|---|
| Assets | 6 | BTC, ETH, SOL, XRP, DOGE, ZEC |
| Markets | 8 | USD, EUR, GBP, CNY, JPY, CHF, CAD, AUD |
| **Total calls/run** | **48** | Full re-fetch; dedup handles duplicates |
| Time at 70 req/min | ~41 seconds | |

---

## 3. Output file layout

```
raw/Alpha vantage/crypto/
├── crypto_daily.parquet           ← all 48 pairs; dedupe on (symbol, market, date)
├── pull_manifest_crypto.json      ← per-run summary
├── pull_call_log.jsonl            ← per-call structured log
└── _crypto_dates_baseline.json    ← written by quality check for stale-delta check
```

### `crypto_daily.parquet` schema

| Column | Type | Source | Notes |
|---|---|---|---|
| `symbol` | str | Meta Data `"2. Digital Currency Code"` | e.g. `"BTC"` |
| `market` | str | Meta Data `"4. Market Code"` | e.g. `"USD"` |
| `date` | date | time series key | |
| `open` | float | `"1. open"` | Quoted string in response |
| `high` | float | `"2. high"` | |
| `low` | float | `"3. low"` | |
| `close` | float | `"4. close"` | |
| `volume` | float | `"5. volume"` | Quoted string in response |
| `pulled_at` | timestamptz | runtime | UTC fetch time |

Dedupe key: `(symbol, market, date)`.

### `pull_call_log.jsonl` — per-call log

| Field | Values |
|---|---|
| `ts` | ISO UTC timestamp |
| `endpoint` | `"DIGITAL_CURRENCY_DAILY"` |
| `key_param` | `"BTC/USD"` etc. |
| `status` | `ok` / `no_data` / `rate_limit` / `subscription_error` / `http_error` |
| `rows` | int |
| `elapsed_s` | float |
| `av_message` | str (optional) |
| `note` | str (optional, redacted) |

Run-boundary sentinels (`run_start` / `run_end` / `run_interrupted`) as `"type": "run_event"` entries.

---

## 4. Endpoint call pattern

```
GET https://www.alphavantage.co/query
  ?function=DIGITAL_CURRENCY_DAILY
  &symbol=BTC
  &market=USD          ← iterate over 8 markets per asset
  &apikey=<key>
```

Response: `{"Meta Data": {...}, "Time Series (Digital Currency Daily)": {"YYYY-MM-DD": {"1. open": "...", ...}}}`.
All OHLCV values are quoted strings. Volume is also a quoted string (confirmed from example).

---

## 5. How to run

```bash
cd /Users/karychen/project/Kairosys

# Full pull (~41 seconds)
python3 scripts/pull_av_crypto.py 2>&1 | tee "raw/Alpha vantage/crypto/pull.log"

# Quality check (run after pull)
python3 scripts/check_av_crypto_coverage.py
```

Wired to SessionStart with `--if-stale 20` — runs automatically each session.

---

## 6. Data quality checks — `scripts/check_av_crypto_coverage.py`

| # | Check | Behaviour |
|---|---|---|
| 1 | All 48 pairs present | FAIL if any missing |
| 2 | History depth per symbol (USD) | FAIL if first date after threshold |
| 3 | Value completeness | WARN if null rate > 5%; never fail |
| 4 | Value spike alert (close p99 × 1.30) | WARN only; never fail |
| 5 | Frequency consistency (~365 rows/year) | FAIL if any year < 300 obs |
| 6 | Recency — crypto trades every day | FAIL if latest date > 3 days ago |
| 7 | Stale-data delta (baseline comparison) | WARN if unchanged > 60 days |

---

## 7. Known gaps

| Item | Status | Reason |
|---|---|---|
| BNB, TRX | Not available | AV returns `"Invalid API call"` for these symbols |
| SOL < 10 years | By design | SOL launched April 2020; max depth is 6 years |
| Crypto intraday | Not available | AV's `CRYPTO_INTRADAY` ignores `month=`; only ~350 bars (~1 day) returned |
| Pre-2010 BTC | Not available | AV's earliest BTC data is 2010-07-17 |
