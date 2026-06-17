# -*- coding: utf-8 -*-
import os
import pandas as pd
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_stock_long import PenZoneStockLongStrategy
from czsc.connectors.research import get_raw_bars

SDT = "20210101"
EDT = "20230101"
ROOT_TAG = "_bt_pen_zone_etf5_no_atr_20210101_20230101_5m"
root = os.path.join(r"d:\pywork\czsc\examples", ROOT_TAG)
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
        print("\n========== 回测汇总 ==========")
        for k, v in stats.items():
            print(f"{k}: {v}")
    else:
        print("\n没有回测结果，请检查是否生成了 pairs 文件。")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"汇总统计失败: {e}")
