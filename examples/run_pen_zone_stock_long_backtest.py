# -*- coding: utf-8 -*-
"""PenZoneStockLongStrategy 回测脚本（5分钟K，MA250，14:30~15:00 买入窗）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")

from czsc.connectors.research import get_raw_bars, get_symbols
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_stock_long import PenZoneStockLongStrategy


# ---------- 回测参数（按需修改）----------
SDT = "20190101"
EDT = "20231231"
SYMBOL_GROUP = "A股场内基金"  # ETF
MAX_SYMBOLS = 1  # 快速测试
N_JOBS = 1  # 快速测试
# ----------------------------------------

ROOT_TAG = f"_bt_test_visualize"


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
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

    symbols = get_symbols(SYMBOL_GROUP)
    if not symbols:
        raise RuntimeError(f"未找到分组 {SYMBOL_GROUP} 的标的，请检查 czsc_research_cache")
    if MAX_SYMBOLS:
        symbols = symbols[: int(MAX_SYMBOLS)]

    print(f"回测: {SYMBOL_GROUP} | {SDT} ~ {EDT} | 标的数={len(symbols)}")
    print(f"输出: {results_path}")
    dummy.execute(symbols, n_jobs=N_JOBS)

    pos_name = "PenZoneStockLongV1"
    try:
        stats = dummy.one_pos_stats(pos_name)
        print("\n========== 回测汇总 ==========")
        print(stats)
    except Exception as e:
        print(f"汇总统计跳过: {e}")


if __name__ == "__main__":
    main()
