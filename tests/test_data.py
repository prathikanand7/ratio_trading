"""
Tests for the Binance Vision parser, run against locally built archives in
Binance's exact format.

Why bother: the parser has to cope with two silent-failure quirks in the real
archive. Older monthly files have no header row, newer ones do, and if you
guess wrong the header becomes a data row and the first bar is garbage.
Separately, Binance switched close_time from milliseconds to microseconds
partway through, and misreading that puts your bars in 1970 or the year
57000. Neither failure raises an exception, so both need a test.
"""

import io
import zipfile

import numpy as np
import pandas as pd

from core.data import _parse_kline_zip, _months_back, KLINE_COLS


def make_archive(n=48, with_header=False, unit="ms", start_ms=1_700_000_000_000):
    """Build a zipped CSV shaped exactly like a Binance Vision monthly file."""
    rows = []
    for i in range(n):
        ot = start_ms + i * 3_600_000
        ct = ot + 3_600_000 - 1
        if unit == "us":
            ot, ct = ot * 1000, ct * 1000
        px = 3000 + i
        rows.append([ot, px, px + 5, px - 5, px + 1, 10.5,
                     ct, 31500.0, 42, 5.2, 15600.0, 0])

    df = pd.DataFrame(rows, columns=KLINE_COLS)
    csv = df.to_csv(index=False, header=with_header)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ETHUSDT-1h-2025-03.csv", csv)
    return buf.getvalue()


def test_headerless():
    df = _parse_kline_zip(make_archive(n=48, with_header=False))
    assert len(df) == 48, f"expected 48 bars, got {len(df)}"
    assert df["close"].iloc[0] == 3001, df["close"].iloc[0]
    assert df["ts"].dt.year.iloc[0] == 2023, df["ts"].iloc[0]
    print(f"  headerless     : {len(df)} bars, first close {df['close'].iloc[0]}, "
          f"first ts {df['ts'].iloc[0]}")


def test_with_header():
    df = _parse_kline_zip(make_archive(n=48, with_header=True))
    assert len(df) == 48, f"header row leaked into data: {len(df)} rows"
    assert df["close"].iloc[0] == 3001, df["close"].iloc[0]
    # The real tell that a header leaked: the column comes back as object,
    # because the literal string "close" sits in row 0 alongside the numbers.
    assert df["close"].dtype.kind in "fi", f"close parsed as {df['close'].dtype}"
    print(f"  with header    : {len(df)} bars, close dtype {df['close'].dtype}")


def test_microseconds():
    df_ms = _parse_kline_zip(make_archive(n=24, unit="ms"))
    df_us = _parse_kline_zip(make_archive(n=24, unit="us"))
    assert df_ms["ts"].iloc[0] == df_us["ts"].iloc[0], (
        f"unit detection failed: ms={df_ms['ts'].iloc[0]} us={df_us['ts'].iloc[0]}")
    print(f"  us vs ms       : both resolve to {df_ms['ts'].iloc[0]}")


def test_hourly_spacing():
    df = _parse_kline_zip(make_archive(n=72))
    gaps = df["ts"].diff().dropna().unique()
    assert len(gaps) == 1, f"irregular spacing: {gaps}"
    assert gaps[0] == pd.Timedelta(hours=1), gaps[0]
    print(f"  bar spacing    : uniform {gaps[0]}")


def test_months_back():
    ms = _months_back(14)
    assert len(ms) == 14
    assert ms == sorted(ms), "months should be chronological"
    today = pd.Timestamp.now("UTC")
    assert (ms[-1][0], ms[-1][1]) != (today.year, today.month), (
        "current month must be skipped: Vision has not published it yet")
    # no duplicates, and the rollover across a year boundary is correct
    assert len(set(ms)) == 14, "duplicate months"
    print(f"  month list     : {ms[0]} .. {ms[-1]} (current month excluded)")


def test_inner_join_drops_unmatched():
    """A bar present in one leg but not the other must not survive."""
    idx_a = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    idx_b = idx_a.delete(5)
    a = pd.DataFrame({"close": np.arange(10.0)}, index=idx_a)
    b = pd.DataFrame({"close": np.arange(9.0)}, index=idx_b)
    idx = a.index.intersection(b.index)
    assert len(idx) == 9, len(idx)
    assert idx_a[5] not in idx
    print(f"  inner join     : 10 and 9 bars -> {len(idx)} matched")


if __name__ == "__main__":
    print("Binance Vision parser tests")
    print("-" * 52)
    for fn in [test_headerless, test_with_header, test_microseconds,
               test_hourly_spacing, test_months_back, test_inner_join_drops_unmatched]:
        fn()
    print("-" * 52)
    print("all passed")
