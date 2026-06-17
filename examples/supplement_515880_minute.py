# -*- coding: utf-8 -*-
"""Supplement 515880.SH intraday parquet for 30m backtest.

Data sources (no CZSC/Tushare token required):
  1. pytdx HQ 5-minute bars (category 0) — history from ~2024-05-31
  2. akshare Sina 1-minute rolling window (~1970 bars, recent days)

Limitations:
  - akshare fund_etf_hist_min_em (Eastmoney) unreachable from this network
  - pytdx 1-minute (category 8) only covers ~2026-01-15 onward
  - True full-history 1-minute needs CZSC cooperation API or Tushare pro_bar_minutes

Usage:
  python examples/supplement_515880_minute.py
  python examples/supplement_515880_minute.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.update_research_cache import _merge_save, _normalize_dt  # noqa: E402

SYMBOL = "515880.SH"
DEFAULT_CACHE = Path(os.environ.get("czsc_research_cache", r"D:\CZSC投研数据"))
GROUP = "A股场内基金"
PYTDX_HOST = "218.75.126.9"
PYTDX_PORT = 7709
INTRADAY_START = pd.Timestamp("2024-05-31")


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
        logger.warning(f"sina 1m failed: {last_err}")
        return pd.DataFrame()

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
    """Keep pre-intraday daily bars; replace overlapping range with intraday data."""
    if old.empty and pytdx_5m.empty and sina_1m.empty:
        return pd.DataFrame()

    base = _normalize_dt(old) if not old.empty else pd.DataFrame(columns=["symbol", "dt", "open", "close", "high", "low", "vol", "amount"])
    intraday_parts = [df for df in (pytdx_5m, sina_1m) if not df.empty]
    if not intraday_parts:
        return base

    intraday = pd.concat(intraday_parts, ignore_index=True)
    intraday = intraday.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)
    cutoff = max(INTRADAY_START, intraday["dt"].min().normalize())
    base = base[base["dt"] < cutoff].copy()
    merged = pd.concat([base, intraday], ignore_index=True)
    merged = merged.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)
    return merged


def verify_get_raw_bars(cache: Path, symbol: str) -> dict:
    os.environ["czsc_research_cache"] = str(cache)
    import czsc  # noqa: WPS433
    from czsc.connectors.research import get_raw_bars  # noqa: WPS433
    from czsc.utils.bar_generator import check_freq_and_market  # noqa: WPS433

    file = cache / GROUP / f"{symbol}.parquet"
    kline = pd.read_parquet(file)
    if "dt" not in kline.columns:
        kline["dt"] = pd.to_datetime(kline["datetime"])
    kline = kline[(kline["dt"] >= "20250101") & (kline["dt"] <= "20251231")]
    uni_times = sorted(kline["dt"].tail(2000).apply(lambda x: x.strftime("%H:%M")).unique().tolist())
    freq_m, market = check_freq_and_market(uni_times, freq="1分钟")

    bars_30 = get_raw_bars(symbol, "30分钟", "20250101", "20251231", fq="前复权")
    bars_2025 = [b for b in bars_30 if b.dt.year == 2025]
    return {
        "unique_times_2025": len(uni_times),
        "freq_market": f"{freq_m}_{market}",
        "bars_30m_2025": len(bars_2025),
        "dt_min_30m": bars_30[0].dt if bars_30 else None,
        "dt_max_30m": bars_30[-1].dt if bars_30 else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplement 515880.SH intraday parquet")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="research cache root")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache = Path(args.cache)
    path = cache / GROUP / f"{args.symbol}.parquet"
    if not path.parent.exists():
        logger.error(f"missing cache group: {path.parent}")
        return 1

    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    old_norm = _normalize_dt(old) if not old.empty else old
    logger.info(f"existing rows={len(old_norm)} unique_times={old_norm['dt'].dt.time.nunique() if not old_norm.empty else 0}")

    logger.info("fetching pytdx 5-minute bars...")
    pytdx_5m = fetch_pytdx_5m(args.symbol)
    logger.info(f"pytdx 5m rows={len(pytdx_5m)} range={pytdx_5m['dt'].min() if len(pytdx_5m) else None} ~ {pytdx_5m['dt'].max() if len(pytdx_5m) else None}")

    logger.info("fetching sina 1-minute rolling window...")
    sina_1m = fetch_sina_1m(args.symbol)
    logger.info(f"sina 1m rows={len(sina_1m)} range={sina_1m['dt'].min() if len(sina_1m) else None} ~ {sina_1m['dt'].max() if len(sina_1m) else None}")

    merged_preview = merge_intraday(old, pytdx_5m, sina_1m)
    n2025 = len(merged_preview[(merged_preview["dt"] >= "2025-01-01") & (merged_preview["dt"] < "2026-01-01")]) if not merged_preview.empty else 0
    logger.info(
        f"merged preview rows={len(merged_preview)} unique_times={merged_preview['dt'].dt.time.nunique() if not merged_preview.empty else 0} "
        f"2025_rows={n2025}"
    )

    if args.dry_run:
        print({"symbol": args.symbol, "merged_rows": len(merged_preview), "2025_rows": n2025, "dry_run": True})
        return 0

    # Replace file content via merge_save: treat merged as 'new' chunk over empty old daily overlap
    keep_pre = merged_preview[merged_preview["dt"] < INTRADAY_START].copy() if not merged_preview.empty else pd.DataFrame()
    intraday_new = merged_preview[merged_preview["dt"] >= INTRADAY_START].copy() if not merged_preview.empty else pd.DataFrame()
    rebuilt = pd.concat([keep_pre, intraday_new], ignore_index=True)
    rebuilt.to_parquet(path, index=False)
    info = {"symbol": args.symbol, "total_rows": len(rebuilt), "max": rebuilt["dt"].max(), "2025_rows": n2025}
    logger.info(f"saved {path}: {info}")

    verify = verify_get_raw_bars(cache, args.symbol)
    print("\n=== verify get_raw_bars('30分钟') 2025 ===")
    for k, v in verify.items():
        print(f"  {k}: {v}")
    if verify.get("unique_times_2025", 0) < 2 or verify.get("bars_30m_2025", 0) < 2:
        print("\n*** verify FAILED: still insufficient intraday data for 30m resample ***")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
