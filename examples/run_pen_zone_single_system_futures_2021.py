# -*- coding: utf-8 -*-
"""
回测：期货主力（2021）- PenZoneSingleSystemMt5Strategy

用法：
python examples/run_pen_zone_single_system_futures_2021.py
"""

import sys
import os
import shutil

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from czsc.connectors.research import get_raw_bars, get_symbols
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone_a0 import PenZoneSingleSystemMt5Strategy


def main():
    # 建议使用项目内路径，避免历史残留目录导致 DummyBacktest 跳过但又无 pairs 文件
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "_bt_pen_zone_single_system_futures_2021"))
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results_20210101_20211231")

    # 如需重跑，建议清理旧结果目录
    clean = True
    if clean and os.path.exists(root):
        shutil.rmtree(root, ignore_errors=True)

    dummy = DummyBacktest(
        strategy=PenZoneSingleSystemMt5Strategy,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt="20210101",
        edt="20211231",
        signals_path=signals_path,
        results_path=results_path,
    )

    symbols = get_symbols("期货主力")
    print(f"期货主力标的数: {len(symbols)}")

    # 先跑 5 个验证链路；确认无异常后可改大或直接全量
    dummy.execute(symbols[:5], n_jobs=1)


if __name__ == "__main__":
    main()

