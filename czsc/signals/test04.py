# from czsc.analyze import CZSC
# from czsc.objects import ZS
# from czsc.utils.sig import create_single_signal
# from collections import OrderedDict
#
#
# class F0Center:
#     """
#     用于存储 f0 级别中枢的区间（zg、zd）及其形成方式
#     formation: "first" 表示首次形成，"up" 表示向上新生，"down" 表示向下新生，
#                "extend" 表示延伸（非新生）。
#     """
#
#     def __init__(self, zg, zd, formation="extend"):
#         self.zg = zg
#         self.zd = zd
#         self.formation = formation
#
#     @property
#     def is_valid(self):
#         return self.zg > self.zd
#
#     def __repr__(self):
#         return f"F0Center(zg={self.zg}, zd={self.zd}, formation={self.formation})"
#
#
# def f0_construct(bi_list, f0_history):
#     """
#     从第一笔开始构造 f0 级别中枢（包含延伸与新生逻辑）：
#       - 遍历所有连续 3 笔，用 ZS(bi_list[i:i+3]) 得到候选中枢区间 [zg, zd]。
#       - 如果当前候选中枢的最后一笔（bi_list[i+2]）满足：
#             回跌候选：其 low >= 当前中枢的 zg  —— 新生，formation="up"
#          或  回升候选：其 high <= 当前中枢的 zd  —— 新生，formation="down"
#         则认为当前中枢结束，新生一个 f0 中枢；否则采用延伸更新（formation="extend"）。
#       - 将所有形成的 f0 中枢记录到 f0_history 中，返回最新的中枢。
#     """
#     if len(bi_list) < 3:
#         return None
#
#     f0_list = []
#     temp_center = None
#
#     for i in range(len(bi_list) - 2):
#         zs = ZS(bi_list[i:i + 3])
#         if not zs.is_valid:
#             continue
#         candidate = F0Center(zg=zs.zg, zd=zs.zd, formation="extend")
#         if temp_center is None:
#             # 第一个有效中枢记为"first"
#             temp_center = F0Center(zg=candidate.zg, zd=candidate.zd, formation="first")
#         else:
#             last_bi = bi_list[i + 2]
#             # 判断是否触发新生
#             if last_bi.low >= temp_center.zg:
#                 # 回跌但不破上轨，向上新生
#                 f0_list.append(temp_center)
#                 temp_center = F0Center(zg=candidate.zg, zd=candidate.zd, formation="up")
#             elif last_bi.high <= temp_center.zd:
#                 # 回升但不破下轨，向下新生
#                 f0_list.append(temp_center)
#                 temp_center = F0Center(zg=candidate.zg, zd=candidate.zd, formation="down")
#             else:
#                 # 延伸：更新中枢区间
#                 temp_center.zg = min(temp_center.zg, candidate.zg)
#                 temp_center.zd = max(temp_center.zd, candidate.zd)
#                 temp_center.formation = "extend"
#     if temp_center:
#         f0_list.append(temp_center)
#
#     f0_history.clear()
#     f0_history.extend(f0_list)
#     return f0_list[-1] if f0_list else None
#
#
# def signal_15m_zs_breakout_new(cat: CZSC, freq="15分钟", **kwargs) -> OrderedDict:
#     """
#    15 分钟级别信号，基于 f0 与 f1 中枢构造：
#
#    参数模板："{freq}_中枢突破_V4"
#
#
#    **信号逻辑：**
#    1. 构造 f0 中枢（至少需要 3 笔形成）；
#    2. 将生成的 f0 中枢累积到历史列表 cat.f0_history 中；
#    3. 当连续 3 个 f0 中枢（即 3 个线段）存在重叠区间，同时价格突破当前 f0 的高点时，触发看多信号；
#    4. 如果形成新的 f1 中枢（即连续 3 个 f0 线段重叠），则作为平仓信号。
#
#    **信号列表：**
#    - Signal('{freq}_中枢突破_V4_看多_任意_任意_0')
#    - Signal('{freq}_中枢突破_V4_平仓_任意_任意_0')
#
#    :param cat: CZSC 对象，表示单一周期的分析对象
#    :param freq: 级别周期（默认 '15分钟'）
#    :param kwargs: 其他参数（例如 di 表示倒数第 di 根 K 线，默认 1）
#    :return: 识别的信号（OrderedDict）
#    """
#
#     di = int(kwargs.get("di", 1))
#     k1, k2, k3 = f"{freq}_中枢突破_V4".split("_")
#     v1 = "其他"
#
#     if not cat or len(cat.bi_list) < 3:
#         return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")
#
#     last_close = cat.bars_raw[-di].close
#
#     if not hasattr(cat, "f0_history"):
#         cat.f0_history = []
#
#     # 构造 f0 中枢（从第一笔开始，包含延伸与新生逻辑）
#     new_f0 = f0_construct(cat.bi_list, cat.f0_history)
#     if not new_f0:
#         return create_single_signal(k1=k1, k2=k2, k3=k3, v1="无有效f0")
#
#     # 根据最新 f0 中枢的 formation 决定信号：
#     # 向上新生 f0 中枢开仓；向下新生 f0 中枢平仓
#     if new_f0.formation == "up":
#         v1 = "看多"
#     elif new_f0.formation == "down":
#         v1 = "平仓"
#     else:
#         v1 = "其他"
#
#     # 输出调试信息
#     print(f"最新 f0 中枢：{new_f0}，最新收盘价：{last_close}")
#     return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
