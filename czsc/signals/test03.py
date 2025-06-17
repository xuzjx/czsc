from czsc.analyze import CZSC

from czsc import CzscSignals
from czsc.objects import ZS
from czsc.utils.sig import create_single_signal
from collections import OrderedDict


def signal_15m_zs_breakout(cat: CZSC, freq="15分钟", **kwargs) -> OrderedDict:
    """
    15 分钟级别，形成中枢后，突破 ZG 开多，回到 ZG 或形成新的 15 分钟中枢平仓；贡献者：你的名字

    参数模板："{freq}_中枢突破_V20250312"

    **信号逻辑：**
    1. **开多条件：**
       - `freq`（默认 15 分钟）形成 **中枢**。
       - 最新 K 线 **突破中枢 ZG**，触发开多信号。
    2. **平仓条件：**
       - 价格 **回落至中枢 ZG**，触发平仓信号。
       - **形成新的 15 分钟级别中枢**，触发平仓信号。

    **信号列表：**
    - Signal('{freq}_中枢突破_V20250312_看多_任意_任意_0')
    - Signal('{freq}_中枢突破_V20250312_平仓_任意_任意_0')

    :param cat: CZSC 对象，表示单一周期的分析对象
    :param freq: 级别周期（默认 '15分钟'）。
    :param kwargs: 其他参数（可扩展）。
        - di: 倒数第 `di` 根 K 线，默认 `1`。
    :return: 识别的信号（OrderedDict）。
    """
    di = int(kwargs.get("di", 1))
    k1, k2, k3 = f"{freq}_中枢突破_V20250312".split("_")
    v1 = "其他"

    # **1. 直接从 CZSC 获取 15 分钟数据**
    c1 = cat  # 直接使用 CZSC 对象
    if not c1 or len(c1.bars_raw) < 10:
        print("⚠️ K 线数据不足，无法计算信号")
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    last_close = c1.bars_raw[-1].close  # 最新收盘价

    # **2. 确保 bi_list 可用**
    if len(c1.bi_list) < 7:
        print("⚠️ bi_list 数据不足，无法计算中枢")
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    zs = ZS(c1.bi_list[-7:-1])  # 获取最近的中枢
    if not zs.is_valid:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    # **3. 突破 ZG 则开仓**
    if last_close > zs.zg:
        v1 = "看多"

    # **4. 记录开仓价格（在 cat.cache 中存储）**
    entry_price_key = f"{freq}_entry_price"
    if v1 == "看多":
        cat.cache[entry_price_key] = last_close  # 存储开仓价格

    # **5. 计算平仓条件**
    entry_price = cat.cache.get(entry_price_key, None)

    # **6. 计算是否形成新的 15 分钟中枢**
    recent_bis = c1.bi_list[-14:] if len(c1.bi_list) >= 14 else c1.bi_list
    new_zs = ZS(recent_bis)  # 计算新的中枢
    has_new_zs = new_zs.is_valid  # 如果新中枢有效，则触发平仓

    # **7. 触发平仓**
    if last_close < zs.zg or has_new_zs:
        v1 = "平仓"

    print(f"信号输出：{k1}_{k2}_{k3} = {v1}")
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
