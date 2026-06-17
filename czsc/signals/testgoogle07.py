import numpy as np
import pandas as pd
from typing import List
from czsc import CZSC, Direction
from czsc.traders.base import CzscSignals
from czsc.objects import BI, RawBar, ZS, Signal
from czsc.utils import get_sub_elements, create_single_signal
from czsc.utils.sig import get_zs_seq
from czsc.signals.tas import update_ma_cache, update_macd_cache
from collections import OrderedDict
from deprecated import deprecated
from sklearn.linear_model import LinearRegression


def check_transition_V250420(c: CZSC, **kwargs) -> OrderedDict:
    """中枢演化

    参数模板："{freq}_中枢演化_V250420"

    **信号逻辑：**
    1. 当中枢结构发生变化，判定为新中枢形成；
    2. 在新中枢形成时，若当前价格高于新中枢上沿 zg，则为“新中枢向上突破”；
    3. 若当前价格低于新中枢下沿 zd，则为“新中枢向下破位”；
    4. 若当前价格仍在中枢区间内，则为“新中枢震荡中”；
    5. 除首次识别外，后续信号统一返回“延续中”；
    6. 若无法识别中枢或结构不完整，返回“无中枢”或“中枢不完整”。

    **信号列表：**

    - Signal('15分钟_中枢演化_V250420_新中枢向上突破_任意_任意_0')
    - Signal('15分钟_中枢演化_V250420_新中枢向下破位_任意_任意_0')
    - Signal('15分钟_中枢演化_V250420_延续中_任意_任意_0')
    - Signal('15分钟_中枢演化_V250420_无中枢_任意_任意_0')
    - Signal('15分钟_中枢演化_V250420_中枢不完整_任意_任意_0')

    :param c: CZSC 对象，包含笔序列与 K 线数据
    :param kwargs: 预留参数，当前未使用
    :return: 中枢结构信号结果（OrderedDict）
    """

    freq = c.freq.value
    k1, k2, k3 = f"{freq}", "中枢演化", "V250420"
    v1, v2 = "其他", "任意"

    # Step 1: 获取中枢序列
    zs_list = get_zs_seq(c.bi_list)
    if not zs_list or len(zs_list) < 2:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="无中枢")

    a0 = zs_list[-1]  # 当前中枢
    a1 = zs_list[-2]  # 前一个中枢

    zg = getattr(a0, "zg", getattr(a0, "gg", None))
    zd = getattr(a0, "zd", getattr(a0, "dd", None))
    if zg is None or zd is None:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="中枢不完整")

    # Step 2: 判断是否首次识别新中枢（用中枢起始笔时间作为唯一ID）
    if not hasattr(c, "latest_zs_id"):
        c.latest_zs_id = None

    is_new_zs = a0.bis != a1.bis and a0.bis[0].edt != c.latest_zs_id
    if is_new_zs:
        # 新中枢的起始笔
        first_bi = a0.bis[0]

        if first_bi.direction == Direction.Down and first_bi.low > a1.zg:
            v1 = "新中枢向上突破"
        elif first_bi.direction == Direction.Up and first_bi.high < a1.zd:
            v1 = "新中枢向下破位"
        else:
            v1 = "延续中"  # 极少数容错情况

        c.latest_zs_id = a0.bis[0].edt  # <--- 添加这一行来更新状态

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)




def ma_base_V240601(c: CZSC, **kwargs) -> OrderedDict:
    di = int(kwargs.get("di", 1))
    ma_type = kwargs.get("ma_type", "SMA").upper()
    timeperiod = int(kwargs.get("timeperiod", 5))
    freq = c.freq.value
    k1, k2, k3 = f"{freq}_D{di}{ma_type}#{timeperiod}_分类V240601".split("_")
    key = update_ma_cache(c, ma_type=ma_type, timeperiod=timeperiod)
    bars = get_sub_elements(c.bars_raw, di=di, n=3)
    v1 = "多头" if bars[-1].close >= bars[-1].cache[key] else "空头"
    # v2 = "向上" if bars[-1].cache[key] >= bars[-2].cache[key] else "向下"
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)