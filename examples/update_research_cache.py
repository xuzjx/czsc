# -*- coding: utf-8 -*-
"""Incrementally update local CZSC research parquet cache.

Sources (auto order):
  1. cooperation API (zbczsc.com) when CZSC_TOKEN or ~/.md5.txt token exists
  2. akshare (free fallback) — ETF/futures daily bars at 15:00 as synthetic 1m end-of-day bars

Usage:
  python examples/update_research_cache.py --groups etf,futures --sdt 20230103
  python examples/update_research_cache.py --source akshare --groups etf --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

DEFAULT_CACHE = os.environ.get("czsc_research_cache", r"D:\CZSC投研数据")
GROUP_ETF = "A股场内基金"
GROUP_FUT = "期货主力"
COOP_URL = "http://zbczsc.com:9106"


def _read_token(url: str) -> Optional[str]:
    token = os.getenv("CZSC_TOKEN") if "zbczsc" in url else os.getenv("TUSHARE_TOKEN")
    if token:
        return token.strip()
    hash_key = hashlib.md5(url.encode("utf-8")).hexdigest()
    path = Path.home() / f"{hash_key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _cooperation_available() -> bool:
    token = _read_token(COOP_URL)
    if not token:
        return False
    try:
        import requests

        r = requests.post(
            COOP_URL,
            json={
                "api_name": "etf_basic",
                "token": token,
                "params": {"v": 2, "fields": "code,name"},
                "fields": "",
            },
            timeout=15,
        )
        return r.ok and r.json().get("code") == 0
    except Exception as exc:
        logger.warning(f"cooperation API unreachable: {exc}")
        return False


def _dt_col(df: pd.DataFrame) -> str:
    if "dt" in df.columns:
        return "dt"
    if "datetime" in df.columns:
        return "datetime"
    raise ValueError(f"no datetime column in {df.columns.tolist()}")


def _normalize_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = _dt_col(out)
    out["dt"] = pd.to_datetime(out[col])
    if col != "dt":
        out = out.drop(columns=[col])
    return out


def _gap_start(existing: pd.DataFrame, sdt: str) -> pd.Timestamp:
    floor = pd.to_datetime(sdt)
    if existing.empty:
        return floor
    last = _normalize_dt(existing)["dt"].max()
    return max(floor, last + pd.Timedelta(minutes=1))


def _merge_save(path: Path, old: pd.DataFrame, new: pd.DataFrame, dry_run: bool) -> dict:
    if new.empty:
        return {"symbol": path.stem, "added": 0, "max": None, "skipped": "no new rows"}

    if old.empty:
        merged = _normalize_dt(new).drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)
        added = len(merged)
        if dry_run:
            return {"symbol": path.stem, "added": added, "max": merged["dt"].max(), "dry_run": True}
        keep = ["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]
        merged[keep].to_parquet(path, index=False)
        return {"symbol": path.stem, "added": added, "max": merged["dt"].max()}

    old_norm = _normalize_dt(old)
    new_norm = _normalize_dt(new)
    merged = pd.concat([old_norm, new_norm], ignore_index=True)
    merged = merged.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)

    added = len(merged) - len(old_norm)
    if added <= 0:
        return {"symbol": path.stem, "added": 0, "max": merged["dt"].max(), "skipped": "already up to date"}

    if dry_run:
        return {"symbol": path.stem, "added": added, "max": merged["dt"].max(), "dry_run": True}

    # preserve original column order / extra columns for futures
    if "datetime" in old.columns:
        merged["datetime"] = merged["dt"]
        merged = merged.drop(columns=["dt"])
        for col in old.columns:
            if col not in merged.columns:
                merged[col] = old[col].iloc[0] if len(old) else None
        merged = merged[old.columns]
    else:
        keep = ["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]
        merged = merged[keep]

    merged.to_parquet(path, index=False)
    mx = merged["datetime"].max() if "datetime" in merged.columns else merged["dt"].max()
    return {"symbol": path.stem, "added": added, "max": mx}


def czsc_fut_to_sina(code: str) -> str:
    return code[2:-4].lower() + "0"


def _etf_sina_symbol(symbol: str) -> str:
    code, exch = symbol.split(".")
    return ("sh" if exch.upper() == "SH" else "sz") + code


def _etf_daily_from_sina(symbol: str, sdt: str, edt: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.fund_etf_hist_sina(symbol=_etf_sina_symbol(symbol))
    df = raw.rename(columns={"date": "dt", "volume": "vol"})
    sdt_ts, edt_ts = pd.to_datetime(sdt), pd.to_datetime(edt) + pd.Timedelta(hours=23, minutes=59)
    df["dt"] = pd.to_datetime(df["dt"]) + pd.Timedelta(hours=15)
    df = df[(df["dt"] >= sdt_ts) & (df["dt"] <= edt_ts)].copy()
    df["symbol"] = symbol
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype("int64")
    return df[["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]]


def fetch_etf_daily_akshare(symbol: str, sdt: str, edt: str, retries: int = 3) -> pd.DataFrame:
    import akshare as ak

    code = symbol.split(".")[0]
    sdt_s = pd.to_datetime(sdt).strftime("%Y%m%d")
    edt_s = pd.to_datetime(edt).strftime("%Y%m%d")
    last_err = None
    for i in range(retries):
        try:
            raw = ak.fund_etf_hist_em(
                symbol=code, period="daily", start_date=sdt_s, end_date=edt_s, adjust="hfq"
            )
            col_map = {
                "日期": "dt",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "vol",
                "成交额": "amount",
            }
            df = raw.rename(columns=col_map)
            df["dt"] = pd.to_datetime(df["dt"]) + pd.Timedelta(hours=15)
            df["symbol"] = symbol
            df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype("int64")
            return df[["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]]
        except Exception as exc:
            last_err = exc
            time.sleep(2 * (i + 1))

    logger.warning(f"{symbol}: fund_etf_hist_em failed ({last_err}), fallback to sina")
    return _etf_daily_from_sina(symbol, sdt, edt)


def fetch_fut_daily_akshare(symbol: str, sdt: str, edt: str, retries: int = 3) -> pd.DataFrame:
    import akshare as ak

    sina = czsc_fut_to_sina(symbol)
    last_err = None
    for i in range(retries):
        try:
            raw = ak.futures_zh_daily_sina(symbol=sina)
            break
        except Exception as exc:
            last_err = exc
            time.sleep(2 * (i + 1))
    else:
        raise last_err

    df = raw.rename(columns={"date": "dt", "volume": "vol"})
    df["dt"] = pd.to_datetime(df["dt"]) + pd.Timedelta(hours=15)
    sdt_ts, edt_ts = pd.to_datetime(sdt), pd.to_datetime(edt) + pd.Timedelta(hours=23, minutes=59)
    df = df[(df["dt"] >= sdt_ts) & (df["dt"] <= edt_ts)].copy()
    df["symbol"] = symbol
    df["contract"] = symbol
    df["amount"] = (df["vol"] * df["close"]).astype("float64")
    if "factor" not in df.columns:
        df["factor"] = 1.0
    return df


def fetch_etf_cooperation(symbol: str, sdt: str, edt: str) -> pd.DataFrame:
    from czsc.connectors import cooperation as coo

    df = coo.get_raw_bars(symbol=symbol, freq="1分钟", sdt=sdt, edt=edt, fq="后复权", raw_bars=False)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        rows = [
            {
                "symbol": b.symbol,
                "dt": b.dt,
                "open": b.open,
                "close": b.close,
                "high": b.high,
                "low": b.low,
                "vol": b.vol,
                "amount": b.amount,
            }
            for b in df
        ]
        df = pd.DataFrame(rows)
    return df


def fetch_fut_cooperation(symbol: str, sdt: str, edt: str) -> pd.DataFrame:
    from czsc.connectors import cooperation as coo

    df = coo.get_raw_bars(symbol=symbol, freq="1分钟", sdt=sdt, edt=edt, fq="后复权", raw_bars=False)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        rows = [
            {
                "open": b.open,
                "close": b.close,
                "high": b.high,
                "low": b.low,
                "vol": b.vol,
                "amount": b.amount,
                "factor": 1.0,
                "symbol": b.symbol,
                "contract": b.symbol,
                "datetime": b.dt,
            }
            for b in df
        ]
        df = pd.DataFrame(rows)
    return df


def update_group(
    cache: Path,
    group: str,
    source: str,
    sdt: str,
    edt: str,
    dry_run: bool,
    sleep_sec: float,
) -> list[dict]:
    group_path = cache / group
    if not group_path.exists():
        raise FileNotFoundError(group_path)

    use_coop = source == "cooperation" or (source == "auto" and _cooperation_available())
    active_source = "cooperation" if use_coop else "akshare"
    logger.info(f"{group}: using source={active_source}")

    fetch_etf = fetch_etf_cooperation if use_coop else fetch_etf_daily_akshare
    fetch_fut = fetch_fut_cooperation if use_coop else fetch_fut_daily_akshare
    is_etf = group == GROUP_ETF

    results = []
    files = sorted(group_path.glob("*.parquet"))
    for path in tqdm(files, desc=group):
        try:
            old = pd.read_parquet(path)
            start = _gap_start(old, sdt)
            if start > pd.to_datetime(edt):
                results.append({"symbol": path.stem, "added": 0, "max": _normalize_dt(old)["dt"].max(), "skipped": "edt"})
                continue
            start_s = start.strftime("%Y-%m-%d")
            new = fetch_etf(path.stem, start_s, edt) if is_etf else fetch_fut(path.stem, start_s, edt)
            info = _merge_save(path, old, new, dry_run)
            info["source"] = active_source
            results.append(info)
        except Exception as exc:
            logger.error(f"{path.stem}: {exc}")
            results.append({"symbol": path.stem, "error": str(exc)})
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return results


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if r.get("added", 0) > 0]
    err = [r for r in results if "error" in r]
    print(f"\n=== summary: updated={len(ok)} errors={len(err)} total={len(results)} ===")
    if ok:
        mx = max(r["max"] for r in ok if r.get("max") is not None)
        print(f"latest bar among updated: {mx}")
    for r in err[:10]:
        print(f"  ERR {r['symbol']}: {r['error']}")
    if len(err) > 10:
        print(f"  ... and {len(err) - 10} more errors")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update CZSC research parquet cache")
    p.add_argument("--cache", default=DEFAULT_CACHE, help="research cache root")
    p.add_argument("--groups", default="etf,futures", help="etf,futures or comma list")
    p.add_argument("--source", choices=["auto", "cooperation", "akshare"], default="auto")
    p.add_argument("--sdt", default="20230103", help="incremental start (YYYYMMDD)")
    p.add_argument("--edt", default=datetime.now().strftime("%Y%m%d"), help="end date")
    p.add_argument("--dry-run", action="store_true", help="fetch only, do not write parquet")
    p.add_argument("--sleep", type=float, default=0.3, help="seconds between akshare calls")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache = Path(args.cache)
    if not cache.exists():
        logger.error(f"cache not found: {cache}")
        return 1

    group_map = {
        "etf": GROUP_ETF,
        "futures": GROUP_FUT,
        "A股场内基金": GROUP_ETF,
        "期货主力": GROUP_FUT,
    }
    groups = []
    for g in args.groups.split(","):
        g = g.strip()
        if g not in group_map:
            logger.error(f"unknown group: {g}")
            return 1
        groups.append(group_map[g])

    logger.info(f"cache={cache} sdt={args.sdt} edt={args.edt} source={args.source} dry_run={args.dry_run}")
    if args.source != "akshare":
        token = _read_token(COOP_URL)
        logger.info(f"CZSC/cooperation token: {'present' if token else 'missing'}")

    all_results = []
    for group in groups:
        all_results.extend(
            update_group(cache, group, args.source, args.sdt, args.edt, args.dry_run, args.sleep)
        )
    summarize(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
