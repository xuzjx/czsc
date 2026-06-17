# -*- coding: utf-8 -*-
"""Supplement intraday parquet for ALL 指月自选 ETFs (for 30m backtest 2025).

Goal:
  Ensure `czsc.connectors.research.get_raw_bars(symbol, "30分钟")` works for 2025
  by writing intraday bars (pytdx 5m + Sina 1m rolling window) into
  `czsc_research_cache/A股场内基金/<symbol>.parquet`.

Notes:
  - No CZSC_TOKEN / Tushare required.
  - pytdx 5m typically provides history from ~2024-05-31 onwards (enough for 2025).
  - Sina 1m provides only a recent rolling window; used to top-up latest days.

Usage:
  python examples/supplement_zhiyue_etf_minute.py
  python examples/supplement_zhiyue_etf_minute.py --dry-run
  python examples/supplement_zhiyue_etf_minute.py --cache D:\CZSC投研数据
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.update_research_cache import _normalize_dt  # noqa: E402

DEFAULT_CACHE = Path(os.environ.get("czsc_research_cache", r"D:\CZSC投研数据"))
GROUP = "A股场内基金"
ZHIYUE_DIR = _REPO_ROOT / "指月自选etf"
ZHIYUE_CODES = ZHIYUE_DIR / "codes.tsv"

PYTDX_HOST = "218.75.126.9"
PYTDX_PORT = 7709
INTRADAY_START = pd.Timestamp("2024-05-31")


def load_zhiyue_symbols() -> Tuple[List[str], Dict[str, str]]:
    df = pd.read_csv(ZHIYUE_CODES, sep="\t", dtype=str)
    symbols = df["symbol"].dropna().astype(str).tolist()
    name_map = dict(zip(df["symbol"].astype(str), df["name"].astype(str)))
    return symbols, name_map


def _etf_sina_symbol(symbol: str) -> str:
    code, exch = symbol.split(".")
    return ("sh" if exch.upper() == "SH" else "sz") + code


def fetch_pytdx_5m(symbol: str, host: str = PYTDX_HOST, port: int = PYTDX_PORT) -> pd.DataFrame:
    from pytdx.hq import TdxHq_API

    code, exch = symbol.split(".")
    market = 1 if exch.upper() == "SH" else 0
    api = TdxHq_API()
    if not api.connect(host, port, time_out=8):
        raise ConnectionError(f"pytdx connect failed: {host}:{port}")

    rows = []
    start = 0
    batch = 800
    try:
        while True:
            chunk = api.get_security_bars(0, market, code, start, batch)
            if not chunk:
                break
            rows.extend(chunk)
            start += batch
            if start > 100_000:
                break
    finally:
        api.disconnect()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "symbol": symbol,
            "dt": pd.to_datetime([r["datetime"] for r in rows]),
            "open": [r["open"] for r in rows],
            "close": [r["close"] for r in rows],
            "high": [r["high"] for r in rows],
            "low": [r["low"] for r in rows],
            "vol": [int(r["vol"]) for r in rows],
            "amount": [float(r["amount"]) for r in rows],
        }
    )
    return df.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)


def fetch_sina_1m(symbol: str, retries: int = 3) -> pd.DataFrame:
    import akshare as ak

    sina = _etf_sina_symbol(symbol)
    last_err = None
    for i in range(retries):
        try:
            raw = ak.stock_zh_a_minute(symbol=sina, period="1", adjust="qfq")
            break
        except Exception as exc:
            last_err = exc
            time.sleep(2 * (i + 1))
    else:
        raise RuntimeError(f"sina 1m failed: {last_err}")

    df = raw.rename(columns={"day": "dt", "volume": "vol"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["symbol"] = symbol
    for col in ("open", "close", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype("float64")
    keep = ["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]
    return df[keep].drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)


def merge_intraday(old: pd.DataFrame, pytdx_5m: pd.DataFrame, sina_1m: pd.DataFrame) -> pd.DataFrame:
    """Keep pre-intraday daily bars; replace overlapping range with intraday bars."""
    if old.empty and pytdx_5m.empty and sina_1m.empty:
        return pd.DataFrame()

    base = _normalize_dt(old) if not old.empty else pd.DataFrame(
        columns=["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]
    )
    parts = [df for df in (pytdx_5m, sina_1m) if not df.empty]
    if not parts:
        return base

    intraday = pd.concat(parts, ignore_index=True)
    intraday = intraday.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)
    cutoff = max(INTRADAY_START, intraday["dt"].min().normalize())

    base = base[base["dt"] < cutoff].copy()
    merged = pd.concat([base, intraday], ignore_index=True)
    merged = merged.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)
    return merged


def _verify_get_raw_bars_30m_2025(cache: Path, symbol: str) -> Dict:
    os.environ["czsc_research_cache"] = str(cache)
    from czsc.connectors.research import get_raw_bars  # noqa: WPS433
    from czsc.utils.bar_generator import check_freq_and_market  # noqa: WPS433

    file = cache / GROUP / f"{symbol}.parquet"
    kline = pd.read_parquet(file)
    if "dt" not in kline.columns and "datetime" in kline.columns:
        kline["dt"] = pd.to_datetime(kline["datetime"])
    kline["dt"] = pd.to_datetime(kline["dt"], errors="coerce")
    k2025 = kline[(kline["dt"] >= "2025-01-01") & (kline["dt"] <= "2025-12-31 23:59:59")].copy()
    uni_times = (
        sorted(k2025["dt"].tail(2000).dt.strftime("%H:%M").dropna().unique().tolist()) if not k2025.empty else []
    )
    freq_m, market = check_freq_and_market(uni_times, freq="1分钟") if uni_times else ("unknown", "unknown")

    bars_30 = get_raw_bars(symbol, "30分钟", "20250101", "20251231", fq="前复权")
    bars_2025 = [b for b in bars_30 if b.dt.year == 2025]
    return {
        "unique_times_2025": int(len(uni_times)),
        "freq_market": f"{freq_m}_{market}",
        "bars_30m_2025": int(len(bars_2025)),
        "dt_min_30m": bars_30[0].dt if bars_30 else None,
        "dt_max_30m": bars_30[-1].dt if bars_30 else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplement 指月自选 ETF intraday parquet cache")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="research cache root")
    p.add_argument("--dry-run", action="store_true", help="fetch & preview only, do not write parquet")
    p.add_argument("--sleep", type=float, default=0.15, help="sleep seconds between symbols")
    p.add_argument("--pytdx-host", default=PYTDX_HOST)
    p.add_argument("--pytdx-port", type=int, default=PYTDX_PORT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache = Path(args.cache)
    group_path = cache / GROUP
    if not group_path.exists():
        logger.error(f"missing cache group: {group_path}")
        return 1

    symbols, name_map = load_zhiyue_symbols()
    logger.info(f"loaded symbols={len(symbols)} from {ZHIYUE_CODES}")

    out_rows = []
    ok = 0
    failed = 0
    for i, symbol in enumerate(symbols, start=1):
        name = name_map.get(symbol, "")
        path = group_path / f"{symbol}.parquet"
        row = {"symbol": symbol, "name": name, "path": str(path)}
        try:
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            old_norm = _normalize_dt(old) if not old.empty else old
            row["old_rows"] = int(len(old_norm))
            row["old_unique_times_tail2000"] = int(old_norm["dt"].tail(2000).dt.strftime("%H:%M").nunique()) if not old_norm.empty else 0

            pytdx_5m = fetch_pytdx_5m(symbol, host=args.pytdx_host, port=args.pytdx_port)
            sina_1m = pd.DataFrame()
            try:
                sina_1m = fetch_sina_1m(symbol)
            except Exception as exc:
                row["sina_1m_error"] = str(exc)

            row["pytdx_5m_rows"] = int(len(pytdx_5m))
            row["sina_1m_rows"] = int(len(sina_1m))
            row["pytdx_5m_min"] = "" if pytdx_5m.empty else str(pytdx_5m["dt"].min())
            row["pytdx_5m_max"] = "" if pytdx_5m.empty else str(pytdx_5m["dt"].max())
            row["sina_1m_min"] = "" if sina_1m.empty else str(sina_1m["dt"].min())
            row["sina_1m_max"] = "" if sina_1m.empty else str(sina_1m["dt"].max())

            merged = merge_intraday(old, pytdx_5m, sina_1m)
            row["merged_rows"] = int(len(merged))
            row["merged_min"] = "" if merged.empty else str(merged["dt"].min())
            row["merged_max"] = "" if merged.empty else str(merged["dt"].max())
            n2025 = int(len(merged[(merged["dt"] >= "2025-01-01") & (merged["dt"] < "2026-01-01")])) if not merged.empty else 0
            row["merged_rows_2025"] = n2025
            row["merged_unique_times_2025_tail2000"] = (
                int(merged[(merged["dt"] >= "2025-01-01") & (merged["dt"] < "2026-01-01")]["dt"].tail(2000).dt.strftime("%H:%M").nunique())
                if not merged.empty
                else 0
            )

            if not args.dry_run:
                keep = ["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]
                merged[keep].to_parquet(path, index=False)

            verify = _verify_get_raw_bars_30m_2025(cache, symbol)
            row.update(verify)
            row["ok_30m_2025"] = int(verify.get("unique_times_2025", 0) >= 2 and verify.get("bars_30m_2025", 0) >= 2)

            ok += int(row["ok_30m_2025"])
        except Exception as exc:
            failed += 1
            row["error"] = str(exc)
            row["ok_30m_2025"] = 0

        out_rows.append(row)
        if i % 5 == 0:
            logger.info(f"progress {i}/{len(symbols)} ok={ok} failed={failed}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    out_df = pd.DataFrame(out_rows)
    out_dir = _REPO_ROOT / "examples" / "_portfolio_nav_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "zhiyue_minute_cache_status_2025.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"saved report: {out_csv} rows={len(out_df)} ok={ok} failed={failed}")

    # console summary
    print("\n=== supplement summary ===")
    print(f"symbols={len(symbols)} ok_30m_2025={ok} failed={failed} dry_run={args.dry_run}")
    if failed:
        bad = out_df.loc[out_df.get("error").notna(), ["symbol", "name", "error"]].head(20)
        print("\nfirst errors:")
        print(bad.to_string(index=False))
    print(f"\nreport -> {out_csv}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

