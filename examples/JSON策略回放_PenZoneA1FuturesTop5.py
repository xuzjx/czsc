# -*- coding: utf-8 -*-
"""PenZoneA1Strategy 期货 Top5 回测（5分钟K，ATR 止损）

从 futures_per_symbol_ann_sharpe.csv 5m 周期按 sharpe 选出的 Top5 品种，
回测区间 20170101~20230101（数据不足的品种从上市日起算）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1Strategy

# ---------- 回测参数 ----------
SDT = "20170101"
EDT = "20230101"
FREQ = "5分钟"
# Top5 by sharpe (5m, 2021-2023 full futures run): 沪铜/棉花/生猪/花生/不锈钢
SYMBOLS = [
    "SQcu9001",  # 沪铜
    "ZZCF9001",  # 棉花
    "DLlh9001",  # 生猪
    "ZZPK9001",  # 花生
    "SQss9001",  # 不锈钢
]
N_JOBS = 5
# --------------------------------

ROOT_TAG = f"_bt_pen_zone_a1_stop_futures_top5_{SDT}_{EDT}_5m"


class PenZoneA1Strategy5m(PenZoneA1Strategy):
    """PenZoneA1 5分钟周期版（ATR 止损内置于信号函数）"""

    base_freq = FREQ


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")

    dummy = DummyBacktest(
        strategy=PenZoneA1Strategy5m,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )

    print(f"回测: PenZoneA1Strategy5m | {SDT} ~ {EDT} | freq={FREQ}")
    print(f"标的: {SYMBOLS}")
    print(f"输出: {results_path}")

    dummy.execute(SYMBOLS, n_jobs=N_JOBS)

    for pos_name in ("PenZoneA1V1",):
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
