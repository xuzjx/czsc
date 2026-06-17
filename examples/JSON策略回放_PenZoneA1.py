# -*- coding: utf-8 -*-
"""PenZoneA1Strategy 回测脚本（15分钟K，A0/A1 中枢突破）

配置对齐 pen_zone etf5 回测：5 只 A 股场内基金 ETF，20210101~20230101。
策略原生周期为 15 分钟（与 PenZoneStockLong 的 5 分钟不同）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars, get_symbols
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1Strategy

# ---------- 回测参数（对齐 etf5 pen_zone 回测）----------
SDT = "20210101"
EDT = "20230101"
# 与 _bt_pen_zone_etf5_20210101_20230101_5m 相同的 5 只 ETF
SYMBOLS = [
    "159901.SZ",
    "159902.SZ",
    "159903.SZ",
    "159905.SZ",
    "159906.SZ",
]
N_JOBS = 1
# --------------------------------------------------------

ROOT_TAG = f"_bt_pen_zone_a1_etf5_{SDT}_{EDT}_15m"


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")

    dummy = DummyBacktest(
        strategy=PenZoneA1Strategy,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )

    symbols = SYMBOLS
    # 若 SYMBOLS 为空则回退到分组
    if not symbols:
        symbols = get_symbols("A股场内基金")
        if not symbols:
            raise RuntimeError("未找到标的，请检查 czsc_research_cache 数据目录")
        symbols = symbols[:5]

    print(f"回测: PenZoneA1Strategy | {SDT} ~ {EDT} | freq=15分钟")
    print(f"标的: {symbols}")
    print(f"输出: {results_path}")

    dummy.execute(symbols, n_jobs=N_JOBS)

    for pos_name in ("PenZoneA1V1", "PenZoneA1AddV1"):
        try:
            stats = dummy.one_pos_stats(pos_name)
            if stats:
                print(f"\n========== {pos_name} 回测汇总 ==========")
                for k, v in stats.items():
                    if k != "pos_dump":
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"{pos_name} 汇总统计跳过: {e}")


if __name__ == "__main__":
    main()
