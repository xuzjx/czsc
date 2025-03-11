from collections import OrderedDict

from czsc import CZSC, Direction
from czsc.utils import create_single_signal


def cxt_zjx_V240910(c: CZSC, **kwargs) -> OrderedDict:
    """一两根K线快速突破反向笔

    参数模板："{freq}_快速突破_BE辅助V230815"

    **信号逻辑：**

    以向上笔为例：右侧分型完成后第一根K线的最低价低于该笔的最低价，认为向上笔结束，反向向下笔开始。

    **信号列表：**

    - Signal('15分钟_快速突破_BE辅助V230815_向下_任意_任意_0')
    - Signal('15分钟_快速突破_BE辅助V230815_向上_任意_任意_0')

    :param c: CZSC对象
    :param kwargs:
    :return: 信号识别结果
    """

    freq = c.freq.value

    k1, k2, k3 = f"{freq}_快速突破_BE辅助V230815".split("_")
    v1 = "其他"
    if len(c.bi_list) < 5 or len(c.bars_ubi) >= 5:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    bi, last_bar = c.bi_list[-1], c.bars_ubi[-1]
    if bi.direction == Direction.Up and last_bar.low < bi.low:
        v1 = "向下"
    if bi.direction == Direction.Down and last_bar.high > bi.high:
        v1 = "向上"
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


def check():
    from czsc.connectors import research
    from czsc.traders.base import check_signals_acc

    # 获取历史k线数据
    symbols = research.get_symbols('A股主要指数')
    bars = research.get_raw_bars(symbols[0], '15分钟', '20181101', '20210101', fq='前复权')

    signals_config = [{'name': cxt_zjx_V240910, 'freq': "60分钟", 'N': 60}]
    check_signals_acc(bars, signals_config=signals_config, height='780px', delta_days=1)  # type: ignore


if __name__ == '__main__':
    check()