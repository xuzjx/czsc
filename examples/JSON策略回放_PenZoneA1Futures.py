# -*- coding: utf-8 -*-
"""PenZoneA1Strategy 期货全品种回测（15分钟K，ATR 止损，对齐实盘 MQL5 参数）

参考：
- czsc/signals/strategies_pen_zone_a1.py（PenZoneA1Strategy：atr_period=14, in_sl=6, base_freq=15分钟）
- 与 _bt_pen_zone_a1_stop_futures_20210101_20230101_15m 输出目录一致

标的：投研数据「期货主力」全部分组（约 56 个连续合约代码）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars, get_symbols
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1Strategy

# ---------- 回测参数（对齐 PenZoneA1Strategy 实盘/MQL5 默认）----------
SDT = "20210101"
EDT = "20230101"
FREQ = "15分钟"
ATR_PERIOD = 14
IN_SL = 6
N_JOBS = 4
# ---------------------------------------------------------------------

ROOT_TAG = f"_bt_pen_zone_a1_stop_futures_{SDT}_{EDT}_15m"


class PenZoneA1StrategyFutures(PenZoneA1Strategy):
    """期货 15 分钟版：继承 PenZoneA1Strategy 默认 atr_period=14, in_sl=6。"""

    base_freq = FREQ


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")

    symbols = get_symbols("期货主力")
    if not symbols:
        raise RuntimeError("未找到标的，请检查 czsc_research_cache 数据目录")

    dummy = DummyBacktest(
        strategy=PenZoneA1StrategyFutures,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )

    print(f"回测: PenZoneA1StrategyFutures | {SDT} ~ {EDT} | freq={FREQ}")
    print(f"参数: atr_period={ATR_PERIOD}, in_sl={IN_SL}")
    print(f"标的数: {len(symbols)}")
    print(f"输出: {results_path}")

    dummy.execute(symbols, n_jobs=N_JOBS)

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
