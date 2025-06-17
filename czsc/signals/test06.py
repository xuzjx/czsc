from collections import OrderedDict

from czsc import Direction
from czsc.analyze import CZSC
from czsc.utils import create_single_signal


def cxt_up_down_signal(c: CZSC, di=1, **kwargs) -> OrderedDict:
    """简单策略：根据最新笔的方向判断买卖信号

    参数模板："{freq}_D{di}_UpDown"

    **信号逻辑：**
    1. 当最新笔（或倒数第 di 笔）的方向为 "up" 时，发出买入信号；
    2. 当方向为 "down" 时，发出卖出信号；
    3. 其他情况返回“其他”。

    **信号列表示例：**
    - Signal('15分钟_D1_UpDown_买入_任意_任意_0')
    - Signal('15分钟_D1_UpDown_卖出_任意_任意_0')
    - Signal('15分钟_D1_UpDown_其他_任意_任意_0')

    :param c: CZSC 对象，包含 K 线数据及笔列表等
    :param di: 从最后一个笔的第几个开始识别（默认 1）
    :param kwargs: 预留参数
    :return: 信号识别结果（OrderedDict）
    """
    di = int(di)
    k1, k2, k3 = f"{c.freq.value}_D{di}_UpDown".split('_')

    # 如果笔数量不足，则返回默认信号
    if not hasattr(c, "bi_list") or len(c.bi_list) < di:
        print("其他")
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    # 取倒数第 di 笔作为参考
    bi = c.bi_list[-di]
    direction = getattr(bi, "direction", None)
    if direction == Direction.Up:
        print("买入")
        v1 = "买入"
    elif direction == Direction.Down:
        print("卖出")
        v1 = "卖出"
    else:
        v1 = "其他"

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
