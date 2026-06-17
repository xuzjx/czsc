# -*- coding: utf-8 -*-
"""PenZoneA1EtfStrategy 全市场 ETF 日线回测 2025（T+1 只做多，ATR 止损）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars, get_symbols
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1EtfStrategy

SDT = "20250101"
EDT = "20260101"
FREQ = "日线"
N_JOBS = 4
ROOT_TAG = f"_bt_pen_zone_a1_stop_etf_all_t1_{SDT}_{EDT}_daily"


class PenZoneA1EtfStrategyDaily(PenZoneA1EtfStrategy):
    base_freq = FREQ


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")

    symbols = get_symbols("A股场内基金")
    if not symbols:
        raise RuntimeError("未找到标的，请检查 czsc_research_cache 数据目录")

    dummy = DummyBacktest(
        strategy=PenZoneA1EtfStrategyDaily,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )

    print(f"回测: PenZoneA1EtfStrategyDaily | {SDT} ~ {EDT} | freq={FREQ} | T+1 long-only")
    print(f"标的数: {len(symbols)}")
    print(f"输出: {results_path}")

    dummy.execute(symbols, n_jobs=N_JOBS)

    for pos_name in ("PenZoneA1EtfV1",):
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
