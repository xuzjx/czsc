# -*- coding: utf-8 -*-
"""PenZoneA1EtfStrategy 回测脚本（30分钟K，A股 ETF 只做多 T+1）

5 只 A 股场内基金 ETF，20210101~20230101。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1EtfStrategy

SDT = "20210101"
EDT = "20230101"
SYMBOLS = [
    "159901.SZ",
    "159902.SZ",
    "159903.SZ",
    "159905.SZ",
    "159906.SZ",
]
N_JOBS = 1
ROOT_TAG = f"_bt_pen_zone_a1_stop_etf5_t1_{SDT}_{EDT}_30m"


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")

    dummy = DummyBacktest(
        strategy=PenZoneA1EtfStrategy,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )

    print(f"回测: PenZoneA1EtfStrategy | {SDT} ~ {EDT} | freq=30分钟 | T+1 long-only")
    print(f"标的: {SYMBOLS}")
    print(f"输出: {results_path}")

    dummy.execute(SYMBOLS, n_jobs=N_JOBS)

    try:
        stats = dummy.one_pos_stats("PenZoneA1EtfV1")
        if stats:
            print("\n========== PenZoneA1EtfV1 回测汇总 ==========")
            for k, v in stats.items():
                if k != "pos_dump":
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"汇总统计跳过: {e}")


if __name__ == "__main__":
    main()
