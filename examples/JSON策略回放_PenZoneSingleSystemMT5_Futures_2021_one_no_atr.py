# -*- coding: utf-8 -*-
"""
回测：期货单品种 2021 - PenZoneSingleSystemMt5Strategy（关闭 2ATR 对冲保本）

运行：
python examples/JSON策略回放_PenZoneSingleSystemMT5_Futures_2021_one_no_atr.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")

from czsc.connectors.research import get_raw_bars  # noqa: E402
from czsc.traders.dummy import DummyBacktest  # noqa: E402
from czsc.signals.strategies_pen_zone_a0 import (  # noqa: E402
    PenZoneSingleSystemMt5Strategy,
    pen_zone_single_system_signal_mt5_v1,
)

SDT = "20210101"
EDT = "20220101"
FREQ = "15分钟"
SYMBOL = "SQrb9001"

ROOT_TAG = f"_bt_pen_zone_single_system_mt5_no_atr_{SYMBOL}_{SDT}_{EDT}_15m"


class OneSymbolPenZoneSingleSystemMt5NoAtrStrategy(PenZoneSingleSystemMt5Strategy):
    base_freq = FREQ

    @property
    def signals_config(self):
        return [
            {"name": pen_zone_single_system_signal_mt5_v1, "freq": self.base_freq, "di": 1, "log": False},
        ]


def main():
    bars = get_raw_bars(SYMBOL, freq=FREQ, sdt=SDT, edt=EDT)
    assert bars, f"no bars: {SYMBOL} {FREQ} {SDT}-{EDT}"

    dummy = DummyBacktest(
        strategy=OneSymbolPenZoneSingleSystemMt5NoAtrStrategy,
        read_bars=get_raw_bars,
        signals_path=os.path.join(os.path.dirname(__file__), ROOT_TAG, "signals"),
        results_path=os.path.join(os.path.dirname(__file__), ROOT_TAG, "results"),
        sdt=SDT,
        edt=EDT,
        n_bars=300,
        base_freq=FREQ,
    )

    print(f"回测(无ATR保本): {SYMBOL} | {SDT} ~ {EDT} | freq={FREQ}")
    print(f"输出: {os.path.join(os.path.dirname(__file__), ROOT_TAG)}")
    dummy.execute([SYMBOL], n_jobs=1)


if __name__ == "__main__":
    main()
