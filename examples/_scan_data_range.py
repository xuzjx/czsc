# -*- coding: utf-8 -*-
"""Scan all parquet files for global min/max dates."""
import os
from pathlib import Path
import pandas as pd

cache = os.environ.get("czsc_research_cache", r"D:\CZSC投研数据")
groups = ["A股场内基金", "期货主力", "A股主要指数", "中证500成分股"]

global_max = None
global_max_file = None
global_min = None

for group in groups:
    p = Path(cache) / group
    if not p.exists():
        continue
    files = list(p.glob("*.parquet"))
    print(f"\n{group}: {len(files)} files")
    group_max = None
    group_max_sym = None
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  skip {f.stem}: {e}")
            continue
        if "dt" not in df.columns:
            if "datetime" in df.columns:
                df["dt"] = pd.to_datetime(df["datetime"])
            else:
                continue
        mx = df["dt"].max()
        mn = df["dt"].min()
        if group_max is None or mx > group_max:
            group_max = mx
            group_max_sym = f.stem
        if global_max is None or mx > global_max:
            global_max = mx
            global_max_file = f"{group}/{f.stem}"
        if global_min is None or mn < global_min:
            global_min = mn
    print(f"  group max: {group_max} ({group_max_sym})")

print(f"\n=== GLOBAL ===")
print(f"  min: {global_min}")
print(f"  max: {global_max}  ({global_max_file})")
print(f"  has_2025: {global_max is not None and global_max.year >= 2025}")
