"""
Check whether 2025 daily bars are complete for local ETF parquet files.

Inputs:
  - D:/pywork/czsc/指月自选etf/*.parquet

Outputs:
  - D:/pywork/czsc/指月自选etf/check_2025_completeness.csv

Notes:
  - Parquet bars are expected to have a 'dt' column (datetime-like). Daily bars typically have dt at 15:00.
  - We normalize dt to date and check missing trading dates within 2025.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd


WATCH_DIR = Path(r"D:\pywork\czsc\指月自选etf")
OUTPUT_CSV = WATCH_DIR / "check_2025_completeness.csv"

Y_SDT = date(2025, 1, 1)
Y_EDT = date(2025, 12, 31)

REFERENCE_SYMBOLS = ("159901.SZ", "510050.SH")


@dataclass
class SymbolCheck:
    symbol: str
    parquet: str
    n_dates_2025: int
    missing_count: int
    missing_dates: str
    min_date_2025: str
    max_date_2025: str
    min_dt_all: str
    max_dt_all: str
    ok: int


def _to_dates_2025(dt_ser: pd.Series) -> set[date]:
    dts = pd.to_datetime(dt_ser, errors="coerce")
    d = dts.dt.date
    return {x for x in d if x is not None and Y_SDT <= x <= Y_EDT}


def _minmax_dt_str(dt_ser: pd.Series) -> tuple[str, str]:
    dts = pd.to_datetime(dt_ser, errors="coerce")
    if dts.empty:
        return "", ""
    mn = dts.min()
    mx = dts.max()
    if pd.isna(mn) or pd.isna(mx):
        return "", ""
    return mn.strftime("%Y-%m-%d %H:%M:%S"), mx.strftime("%Y-%m-%d %H:%M:%S")


def _read_dt(path: Path) -> pd.Series:
    df = pd.read_parquet(path, columns=["dt"])
    if "dt" not in df.columns:
        raise ValueError(f"missing 'dt' column: {path}")
    return df["dt"]


def _pick_expected_calendar(files: dict[str, Path]) -> tuple[str, list[date]]:
    # Prefer a known-liquid reference symbol to define expected 2025 trading dates.
    for sym in REFERENCE_SYMBOLS:
        p = files.get(sym)
        if p and p.exists():
            dates = sorted(_to_dates_2025(_read_dt(p)))
            return sym, dates

    # Fallback: union of all 2025 dates observed.
    all_dates: set[date] = set()
    for p in files.values():
        all_dates |= _to_dates_2025(_read_dt(p))
    dates = sorted(all_dates)
    return "UNION", dates


def _fmt_dates(dates: Iterable[date], limit: int = 50) -> str:
    ds = [x.isoformat() for x in dates]
    if len(ds) <= limit:
        return ",".join(ds)
    head = ds[:limit]
    return ",".join(head) + f",...(+{len(ds) - limit})"


def main() -> int:
    if not WATCH_DIR.exists():
        raise FileNotFoundError(f"watch dir not found: {WATCH_DIR}")

    parquet_files = sorted(WATCH_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under: {WATCH_DIR}")

    files = {p.stem: p for p in parquet_files}  # stem should be symbol like 159901.SZ
    cal_src, expected = _pick_expected_calendar(files)
    expected_set = set(expected)

    rows: list[SymbolCheck] = []
    for sym, p in sorted(files.items()):
        dt_ser = _read_dt(p)
        all_min, all_max = _minmax_dt_str(dt_ser)
        dates_2025 = _to_dates_2025(dt_ser)
        missing = sorted(expected_set - dates_2025)

        min_2025 = min(dates_2025).isoformat() if dates_2025 else ""
        max_2025 = max(dates_2025).isoformat() if dates_2025 else ""
        ok = int(len(missing) == 0 and len(dates_2025) == len(expected))

        rows.append(
            SymbolCheck(
                symbol=sym,
                parquet=str(p),
                n_dates_2025=len(dates_2025),
                missing_count=len(missing),
                missing_dates=_fmt_dates(missing, limit=200),
                min_date_2025=min_2025,
                max_date_2025=max_2025,
                min_dt_all=all_min,
                max_dt_all=all_max,
                ok=ok,
            )
        )

    out = pd.DataFrame([asdict(x) for x in rows]).sort_values(["ok", "missing_count", "symbol"], ascending=[True, False, True])
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    n_total = len(out)
    n_ok = int(out["ok"].sum())
    n_bad = n_total - n_ok
    expected_count = len(expected)
    exp_min = expected[0].isoformat() if expected else ""
    exp_max = expected[-1].isoformat() if expected else ""

    print(f"Watch dir: {WATCH_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Expected calendar source: {cal_src}; expected trading days in 2025: {expected_count}; range: {exp_min} ~ {exp_max}")
    print(f"Symbols checked: {n_total}; complete: {n_ok}; incomplete: {n_bad}")
    if n_bad:
        sample = out[out["ok"] == 0].head(10)[["symbol", "n_dates_2025", "missing_count", "min_date_2025", "max_date_2025"]]
        print("\nTop incomplete symbols (first 10):")
        print(sample.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

