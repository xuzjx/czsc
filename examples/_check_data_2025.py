# -*- coding: utf-8 -*-
"""Quick check of local research data latest bar dates."""
import os
from pathlib import Path
import pandas as pd

cache = os.environ.get("czsc_research_cache", r"D:\CZSC投研数据")
print("cache:", cache, "exists:", Path(cache).exists())
if not Path(cache).exists():
    raise SystemExit(1)

for d in sorted(Path(cache).iterdir()):
    if d.is_dir():
        n = len(list(d.glob("*.parquet")))
        print(f"  {d.name}: {n} parquet files")

samples = {
    "A股场内基金": ["159901.SZ", "510050.SH", "512880.SH"],
    "期货主力": ["DLi9001", "SQcu9001", "SFIF9001"],
}

for group, syms in samples.items():
    p = Path(cache) / group
    if not p.exists():
        print(f"\n=== {group}: NOT FOUND ===")
        continue
    print(f"\n=== {group} ===")
    for sym in syms:
        f = p / f"{sym}.parquet"
        if not f.exists():
            print(f"  {sym}: file missing")
            continue
        df = pd.read_parquet(f)
        dt_col = "dt" if "dt" in df.columns else "datetime"
        if dt_col not in df.columns:
            dt_col = df.columns[0]
        print(f"  {sym}: rows={len(df)}, min={df[dt_col].min()}, max={df[dt_col].max()}")
