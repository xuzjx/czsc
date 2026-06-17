# -*- coding: utf-8 -*-
"""One-off: fetch ETF daily bars into research cache parquet."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.update_research_cache import _merge_save, fetch_etf_daily_akshare

# 默认使用投研缓存环境变量 czsc_research_cache；否则 fallback 到 D:\CZSC投研数据
_cache_root = Path(os.environ.get("czsc_research_cache", r"D:\CZSC投研数据"))
CACHE = _cache_root / "A股场内基金"
SYMBOLS = [
    ("515050.SH", "20191201"),
    ("515880.SH", "20170101"),
]
EDT = datetime.now().strftime("%Y%m%d")


def main() -> int:
    for sym, sdt in SYMBOLS:
        path = CACHE / f"{sym}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        print(f"Fetching {sym} from {sdt} to {EDT}...")
        new = fetch_etf_daily_akshare(sym, sdt, EDT)
        print(f"  fetched {len(new)} rows, {new['dt'].min()} ~ {new['dt'].max()}")
        info = _merge_save(path, old, new, dry_run=False)
        print(f"  saved: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
