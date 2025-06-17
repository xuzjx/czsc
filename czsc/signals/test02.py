from czsc import CZSC, CzscSignals
from czsc.objects import Signal
from collections import OrderedDict

from czsc.utils import create_single_signal


def signal_15m_ma120(c: CZSC, **kwargs) -> OrderedDict:
    """15 分钟级别，价格站上 120 均线做多，跌破 120 均线平仓；贡献者：你的名字

    参数模板："{freq}_MA120趋势_V20250311"

    **信号逻辑：**
    1. **价格 > 60 均线**，做多信号。
    2. **价格 < 60 均线**，平仓信号。

    **信号列表：**
    - Signal('15分钟_MA120趋势_V20250311_看多_任意_任意_0')
    - Signal('15分钟_MA120趋势_V20250311_看空_任意_任意_0')

    :param c: `CZSC` 对象，表示单周期数据
    :param kwargs: 其他参数（可扩展）
    :return: 识别的信号（OrderedDict）
    """
    freq = c.freq.value
    k1, k2, k3 = f"{freq}_MA120趋势_V20250311".split("_")
    v1 = "其他"

    # 确保数据足够
    if len(c.bars_raw) < 60:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    # 计算 60 均线
    closes = [bar.close for bar in c.bars_raw[-60:]]
    ma120 = sum(closes) / 60
    last_close = c.bars_raw[-1].close

    # 判断信号
    if last_close > ma120:
        v1 = "看多"
    elif last_close < ma120:
        v1 = "看空"

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
