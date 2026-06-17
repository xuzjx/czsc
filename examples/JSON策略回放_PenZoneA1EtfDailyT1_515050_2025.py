# -*- coding: utf-8 -*-
"""PenZoneA1EtfStrategy 单标的 ETF 日线回测 2025（T+1 只做多，ATR 止损）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1EtfStrategy

SDT = "20250101"
EDT = "20260101"
FREQ = "日线"
N_JOBS = 1
SYMBOLS = [
    "515050.SH",
    "515880.SH",
]


class PenZoneA1EtfStrategyDaily(PenZoneA1EtfStrategy):
    base_freq = FREQ


def main():
    for symbol in SYMBOLS:
        tag = symbol.replace(".", "")
        root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"_bt_pen_zone_a1_stop_etf_t1_{tag}_{SDT}_{EDT}_daily",
        )
        signals_path = os.path.join(root, "signals")
        results_path = os.path.join(root, "results")

        dummy = DummyBacktest(
            strategy=PenZoneA1EtfStrategyDaily,
            read_bars=get_raw_bars,
            signals_module_name="czsc.signals",
            sdt=SDT,
            edt=EDT,
            signals_path=signals_path,
            results_path=results_path,
        )

        print(f"\n回测: {symbol} | PenZoneA1EtfStrategyDaily | {SDT} ~ {EDT} | freq={FREQ}")
        print(f"输出: {results_path}")

        dummy.execute([symbol], n_jobs=N_JOBS)

        try:
            stats = dummy.one_pos_stats("PenZoneA1EtfV1")
            if stats:
                print(f"\n========== {symbol} PenZoneA1EtfV1 ==========")
                for k, v in stats.items():
                    if k != "pos_dump":
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"汇总统计跳过: {e}")


if __name__ == "__main__":
    main()
