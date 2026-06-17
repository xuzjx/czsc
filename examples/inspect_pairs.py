# -*- coding: utf-8 -*-
import pandas as pd
import os

pairs_file = r"d:\pywork\czsc\examples\_bt_pen_zone_etf5_no_atr_20210101_20230101_5m\results\poss\159901.SZ\PenZoneStockLongV1.pairs"
if os.path.exists(pairs_file):
    df = pd.read_parquet(pairs_file)
    print("Column names:", df.columns.tolist())
    print("First 5 rows of '盈亏比例':")
    print(df['盈亏比例'].head())
    print("Summary of '盈亏比例':")
    print(df['盈亏比例'].describe())
