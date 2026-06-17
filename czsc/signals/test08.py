# import numpy as np
# import pandas as pd
# from typing import List
# from czsc import CZSC
# from czsc.traders.base import CzscSignals
# from czsc.objects import FX, BI, Direction, ZS, Mark
# from czsc.utils import get_sub_elements, create_single_signal
# from czsc.utils.sig import get_zs_seq
# from czsc.signals.tas import update_ma_cache, update_macd_cache
# from collections import OrderedDict
# from deprecated import deprecated
# from sklearn.linear_model import LinearRegression
#
# def cxt_bi_statusV250626(c: CZSC, **kwargs) -> OrderedDict:
#     """笔的表里关系
#
#     参数模板："{freq}_D1_表里关系V250626"
#
#     表里关系的定义参考：http://blog.sina.com.cn/s/blog_486e105c01007wc1.html
#
#     **信号逻辑：**
#
#     1. 最后一笔向下，且未完成笔的长度大于7根K线，表里关系为向上，否则为向下；
#     2. 最后一笔向上，且未完成笔的长度大于7根K线，表里关系为向下，否则为向上；
#     3. 向下的笔遇到底分型，表里关系为底分；向上笔的遇到底分型为延伸；
#     4. 向上的笔遇到顶分型，表里关系为顶分；向下笔的遇到顶分型为延伸。
#
#     **信号列表：**
#
#     - Signal('15分钟_D1_表里关系V250626_向下_底分_任意_0')
#     - Signal('15分钟_D1_表里关系V250626_向下_延伸_任意_0')
#     - Signal('15分钟_D1_表里关系V250626_向上_顶分_任意_0')
#     - Signal('15分钟_D1_表里关系V250626_向上_延伸_任意_0')
#
#     :param c: CZSC 对象
#     :return: 信号字典
#     """
#     print("调试信息", flush=True)  # 立即刷新输出
#
#     k1, k2, k3, v1 = c.freq.value, "D1", "表里关系V250626", "其他"
#     fxs = c.ubi_fxs
#     if len(c.bi_list) < 3 or len(fxs) < 1 or c.bars_raw[-1].dt != fxs[-1].raw_bars[-1].dt:
#         return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
#
#     last_bi = c.bi_list[-1]
#     if last_bi.direction == Direction.Down:
#         v1 = "向上" if len(c.bars_ubi) > 7 else "向下"
#     else:
#         assert last_bi.direction == Direction.Up
#         v1 = "向下" if len(c.bars_ubi) > 7 else "向上"
#
#     if fxs[-1].mark == Mark.D:
#         v2 = "底分" if v1 == "向下" else "延伸"
#     else:
#         assert fxs[-1].mark == Mark.G
#         v2 = "顶分" if v1 == "向上" else "延伸"
#     return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1, v2=v2)