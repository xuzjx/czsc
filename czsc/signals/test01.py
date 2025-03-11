from collections import OrderedDict

from czsc import CzscSignals, Direction
from czsc.utils import create_single_signal


def cxt_15m_zs_3buy_V1(cat: CzscSignals, freq1='15分钟', freq2='1分钟', **kwargs) -> OrderedDict:
    """15 分钟中枢，1 分钟双笔回跌未触及 ZG，形成 3 买信号；贡献者：你的名字

    **信号逻辑：**
    1. 15 分钟级别形成中枢（至少由 3 笔构成）。
    2. 1 分钟级别出现 **双笔回跌**（最近两笔向下）。
    3. 回跌最低点 **未触及** 15 分钟级别的 **中枢上轨（ZG）**。
    4. 满足条件时，返回 **3 买信号**。

    **信号列表：**
    - Signal('15分钟_1分钟_中枢3买V1_3买_任意_任意_0')
    - Signal('15分钟_1分钟_中枢3买V1_未满足_任意_任意_0')

    :param cat: `CzscSignals` 对象
    :param freq1: 大级别周期（默认 '15分钟'）
    :param freq2: 小级别周期（默认 '1分钟'）
    :return: 识别的信号（OrderedDict）
    """
    k1, k2, k3 = f"{freq1}_{freq2}_中枢3买V1".split('_')

    max_freq = cat.kas.get(freq1, None)  # 15分钟
    min_freq = cat.kas.get(freq2, None)  # 1分钟
    symbol = cat.symbol

    # **确保 15 分钟级别计算了笔和中枢**
    max_freq.update_trends()
    # **确保 1 分钟级别计算了笔**
    min_freq.update_trends()

    if not max_freq or not min_freq:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据缺失")

    # 15 分钟级别必须存在中枢
    if len(max_freq.bi_zs) < 1:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="无中枢")

    big_zs = max_freq.bi_zs[-1]  # 最近一个 15 分钟中枢
    zg = big_zs.zg  # 取出中枢上轨 ZG

    # 1 分钟级别最近 2 笔（要求为回跌）
    if len(min_freq.bi_list) < 2:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="无双笔回跌")

    bi1, bi2 = min_freq.bi_list[-2], min_freq.bi_list[-1]

    # 确保是下降笔
    if bi1.direction != Direction.Down or bi2.direction != Direction.Down:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="非双笔回跌")

    # 计算双笔的最低点
    min_low = min(bi1.low, bi2.low)

    # 判断是否形成 3 买
    if min_low >= zg:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="3买")

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1="未满足")
