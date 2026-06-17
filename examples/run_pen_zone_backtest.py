# -*- coding: utf-8 -*-
"""
PenZoneStrategy 回测脚本

说明：
- 使用 DummyBacktest 跑基于笔方向的 PenZoneStrategy 简化版
- 回测结果（pairs / holds / 汇总表）会输出到你指定的目录下
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import os
import czsc
from czsc.connectors.research import get_raw_bars, get_symbols
from czsc.traders.dummy import DummyBacktest
from czsc.signals.strategies_pen_zone import PenZoneStrategy


def main():
    # 回测信号与结果输出路径（建议用环境变量覆盖，避免写死本机路径）
    # 例如：set CZSC_SIGNALS_PATH=D:\czsc_pen_zone\signals
    #      set CZSC_RESULTS_PATH=D:\czsc_pen_zone\results_20200101_20230301
    signals_path = os.environ.get("CZSC_SIGNALS_PATH", r"D:\czsc_pen_zone\signals")
    results_path = os.environ.get("CZSC_RESULTS_PATH", r"D:\czsc_pen_zone\results_20200101_20230301")

    dummy = DummyBacktest(
        strategy=PenZoneStrategy,
        read_bars=get_raw_bars,
        signals_module_name="czsc.signals",
        sdt="20200101",
        edt="20200301",
        signals_path=signals_path,
        results_path=results_path,
    )

    # 选取一组标的进行回测；你可以换成自己的标的列表
    symbols = get_symbols("期货主力")

    # 先只跑一个品种验证链路；确认 OK 后可以把 [:1] 改成 [:10] 或更多
    dummy.execute(symbols[:10], n_jobs=1)


if __name__ == "__main__":
    main()

