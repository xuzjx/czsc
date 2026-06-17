# -*- coding: utf-8 -*-
import os
import pandas as pd
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_stock_long import PenZoneStockLongStrategy
from czsc.connectors.research import get_raw_bars

SDT = "20210101"
EDT = "20230101"

tags = [
    ("_bt_pen_zone_etf5_20210101_20230101_5m", "包含 ATR 过滤"),
    ("_bt_pen_zone_etf5_no_atr_20210101_20230101_5m", "去掉 ATR 过滤")
]

import sys

# 重定向 stdout 到文件
with open(r"d:\pywork\czsc\examples\comparison_final.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    print("========== 策略收益对比 ==========")
    for tag, desc in tags:
        root = os.path.join(r"d:\pywork\czsc\examples", tag)
        signals_path = os.path.join(root, "signals")
        results_path = os.path.join(root, "results")
        
        dummy = DummyBacktest(
            strategy=PenZoneStockLongStrategy,
            read_bars=get_raw_bars,
            signals_module_name="czsc.signals",
            sdt=SDT,
            edt=EDT,
            signals_path=signals_path,
            results_path=results_path,
        )
        
        pos_name = "PenZoneStockLongV1"
        try:
            stats = dummy.one_pos_stats(pos_name)
            if stats:
                print(f"\n模式: {desc}")
                print(f"累计收益 (BP): {stats.get('截面等权累计收益')}")
                print(f"交易胜率: {stats.get('胜率')}")
                print(f"盈亏比: {stats.get('盈亏比')}")
                print(f"交易次数: {stats.get('总交易次数')}")
            else:
                print(f"\n模式: {desc} - 无结果")
        except Exception as e:
            print(f"\n模式: {desc} - 统计失败: {e}")

