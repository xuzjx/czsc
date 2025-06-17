from czsc import CzscSignals
from czsc.objects import Direction, ZS
from czsc.utils.sig import get_sub_elements, create_single_signal
from collections import OrderedDict
import numpy as np


def cxt_15m_zs_3buy_V1(cat: CzscSignals, freq1="15分钟", freq2="1分钟", n=3, **kwargs) -> OrderedDict:
    """
    15 分钟级别，价格站上中枢 3 买信号，跌破 60 均线平仓；贡献者：你的名字

    参数模板："{freq1}_{freq2}_中枢3买V1"

    **信号逻辑：**
    1. **满足条件：**
       - `freq1`（默认 15 分钟）形成 **中枢**。
       - `freq2`（默认 1 分钟）出现 **双笔回跌**（最近两笔为下降笔）。
       - **双笔回跌最低点不低于 `ZG`**。
       - **第 3 笔或 `n` 笔突破双笔最高点**。
    2. **平仓条件：**
       - `freq1`（15 分钟）**收盘价跌破 MA60**，触发退出信号。

    **信号列表：**
    - Signal('{freq1}_{freq2}_中枢3买V1_满足_任意_任意_0')
    - Signal('{freq1}_{freq2}_中枢3买V1_未满足_任意_任意_0')
    - Signal('{freq1}_{freq2}_中枢3买V1_平仓_任意_任意_0')

    :param cat: CzscSignals 对象，包含多周期数据。
    :param freq1: 大级别周期（默认 '15分钟'）。
    :param freq2: 小级别周期（默认 '1分钟'）。
    :param n: 需要突破最高点的笔（默认 `3`）。
    :param kwargs: 其他参数（可扩展）。
        - di: 倒数第 `di` 根 K 线，默认 `1`。
    :return: 识别的信号（OrderedDict）。
    """

    di = int(kwargs.get("di", 1))
    k1, k2, k3 = f"{freq1}_{freq2}_中枢3买V1".split("_")
    v1 = "未满足"

    max_freq = cat.kas.get(freq1, None)
    min_freq = cat.kas.get(freq2, None)

    if not max_freq or not min_freq:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据缺失")

    # **1. 计算 15 分钟级别中枢**
    if len(max_freq.bi_list) < 7:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    bis = get_sub_elements(max_freq.bi_list, di=di, n=7)
    if len(bis) != 7:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    zs = ZS(bis[1:-1])
    if not zs.is_valid:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    # **2. 计算 1 分钟级别的双笔回跌**
    if len(min_freq.bi_list) < n:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    bi1, bi2 = min_freq.bi_list[-2], min_freq.bi_list[-1]
    if bi1.direction != Direction.Down or bi2.direction != Direction.Down:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    # **3. 判断最低点是否触及 ZG**
    min_low = min(bi1.low, bi2.low)
    if min_low < zs.zg:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    # **4. 判断第 n 笔是否突破双笔最高点**
    max_high = max(bi1.high, bi2.high)
    if min_freq.bi_list[-n].high > max_high:
        v1 = "满足"

    # **5. 计算 15 分钟 MA60**
    close_prices = np.array([bar.close for bar in max_freq.bars_raw[-60:]])
    if len(close_prices) < 60:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    ma60 = np.mean(close_prices)  # 计算 MA60
    latest_close = max_freq.bars_raw[-1].close  # 最新收盘价

    # **6. 退出信号：跌破 MA60**
    if latest_close < ma60:
        v1 = "平仓"

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
