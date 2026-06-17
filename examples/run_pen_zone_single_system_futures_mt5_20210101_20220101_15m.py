# -*- coding: utf-8 -*-
"""
回测：期货主力（2021）- PenZoneSingleSystemMt5Strategy（主力 + 对冲）

输出目录：
examples/_bt_pen_zone_single_system_futures_mt5_single_system_20210101_20220101_15m

用法：
python examples/run_pen_zone_single_system_futures_mt5_20210101_20220101_15m.py
"""

from __future__ import annotations

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 若本机已设置 czsc_research_cache，则沿用；否则给一个常见默认值（可按实际环境覆盖）
os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")

from czsc.connectors.research import get_raw_bars, get_symbols  # noqa: E402
from czsc.traders.dummy import DummyBacktest  # noqa: E402
from czsc.signals.strategies_pen_zone_a0 import (  # noqa: E402
    PenZoneSingleSystemMt5Strategy,
    pen_zone_single_system_signal_mt5_v1,
)


SDT = "20210101"
EDT = "20220101"
FREQ = "15分钟"
N_JOBS = 4
CLEAN = False

ROOT_TAG = f"_bt_pen_zone_single_system_futures_mt5_single_system_{SDT}_{EDT}_15m"


class PenZoneSingleSystemMt5StrategyFutures(PenZoneSingleSystemMt5Strategy):
    """期货回测版：关闭 log，避免大量输出拖慢回测。"""

    base_freq = FREQ

    @property
    def signals_config(self):
        return [
            {"name": pen_zone_single_system_signal_mt5_v1, "freq": self.base_freq, "di": 1, "log": False},
        ]


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")

    if CLEAN and os.path.exists(root):
        shutil.rmtree(root, ignore_errors=True)

    symbols = get_symbols("期货主力")
    if not symbols:
        raise RuntimeError("未找到标的，请检查 czsc_research_cache 数据目录")

    dummy = DummyBacktest(
        strategy=PenZoneSingleSystemMt5StrategyFutures,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )

    print(f"回测: PenZoneSingleSystemMt5StrategyFutures | {SDT} ~ {EDT} | freq={FREQ}")
    print(f"标的数: {len(symbols)} | n_jobs={N_JOBS}")
    print(f"输出: {results_path}")

    dummy.execute(symbols, n_jobs=N_JOBS)


if __name__ == "__main__":
    main()

