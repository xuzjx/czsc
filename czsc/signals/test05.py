from datetime import datetime
from czsc import CZSC, CzscStrategyBase
from czsc.utils import get_sub_elements, create_single_signal
from czsc.objects import ZS, Signal, BI
from collections import OrderedDict
from typing import List, Optional


class A0State:
    """中枢状态管理（修正版：基于最近3笔计算重叠区间）"""

    def __init__(self):
        self.active_zs: Optional[ZS] = None  # 当前活跃中枢
        self.frozen_zs: List[ZS] = []          # 已冻结中枢
        self.pending_bis: List[BI] = []        # 候选笔队列（用于初始化中枢）
        self.end_flag: bool = False            # 结束标记（中枢结束时标记为True）
        self.trigger_bi: Optional[BI] = None   # 触发中枢结束的笔

    def update(self, new_bi: BI):
        """更新中枢状态"""
        if not self._validate_bi(new_bi):
            return

        print(f"\n=== [新笔更新] ===")
        print(f"新笔方向：{new_bi.direction}，低点：{new_bi.low}，高点：{new_bi.high}")

        # 如果没有活跃中枢，则收集候选笔
        if not self.active_zs:
            self._process_pending_bis(new_bi)
            return

        # 计算当前中枢的重叠区间（仅使用最近3笔）
        zg, zd = self.get_zs_range()
        if zg is None or zd is None:
            # 重叠区间失效：视为中枢结束
            print("⚠️ 当前中枢重叠区间失效，触发中枢结束")
            self._freeze_active_zs(new_bi)
            return

        # 检查是否满足中枢结束条件
        if self._check_zs_end(new_bi, zg, zd):
            print(f"✅ 中枢结束 @ {new_bi.direction}-笔 | 重叠区间 [ZG={zg:.2f}, ZD={zd:.2f}]")
            self._freeze_active_zs(new_bi)
            return

        # 中枢延续：将新笔加入活跃中枢
        self.active_zs.bis.append(new_bi)
        zg, zd = self.get_zs_range()  # 重新计算最近3笔的区间
        if zg is None or zd is None:
            print("⚠️ 中枢延续后重叠区间失效")
        else:
            print(f"🔄 中枢延续 -> 笔数: {len(self.active_zs.bis)}, 重叠区间 [ZG={zg:.2f}, ZD={zd:.2f}]")

    def get_zs_range(self) -> tuple:
        """
        获取当前活跃中枢的重叠区间 (ZG, ZD)。
        取活跃中枢中最近3笔的high和low：
          ZG = min(highs) ; ZD = max(lows)
        当且仅当 min(highs) > max(lows) 时区间有效，否则返回 (None, None)。
        """
        if not self.active_zs or len(self.active_zs.bis) < 3:
            return None, None
        last_three = self.active_zs.bis[-3:]
        highs = [bi.high for bi in last_three]
        lows = [bi.low for bi in last_three]
        zg = min(highs)
        zd = max(lows)
        if zg <= zd:
            return None, None
        return zg, zd

    def _validate_bi(self, bi: BI) -> bool:
        """验证笔的有效性"""
        if not isinstance(bi, BI):
            raise TypeError("输入必须为BI对象")
        return True

    def _check_zs_end(self, bi: BI, zg: float, zd: float) -> bool:
        """
        中枢结束条件（带动态阈值过滤）：
          对于向下笔：若该笔 low > (当前重叠区间ZG + 阈值)，触发三买；
          对于向上笔：若该笔 high < (当前重叠区间ZD - 阈值)，触发三卖。
        """
        threshold = max(zg * 0.005, 10)  # 动态阈值：0.5%或10点
        if bi.direction == "down":
            return bi.low > zg + threshold
        elif bi.direction == "up":
            return bi.high < zd - threshold
        return False

    def _process_pending_bis(self, bi: BI):
        """处理候选笔队列，尝试初始化中枢"""
        self.pending_bis.append(bi)
        if len(self.pending_bis) >= 3:
            last_three = self.pending_bis[-3:]
            if self._check_overlap(last_three):
                self.active_zs = ZS(bis=last_three)
                # 使用笔的起始时间 (sdt) 作为标识
                print(f"🆕 中枢创建 @ 笔{bi.sdt} | 初始重叠区间 [ZG={self.active_zs.zg:.2f}, ZD={self.active_zs.zd:.2f}]")
                self.pending_bis = []

    def _freeze_active_zs(self, trigger_bi: BI):
        """冻结当前中枢并重置状态，新候选笔以触发笔开始"""
        self.frozen_zs.append(self.active_zs)
        self.end_flag = True
        self.trigger_bi = trigger_bi
        self.active_zs = None
        self.pending_bis = [trigger_bi]
        print(f"⛔ 中枢已冻结，冻结数量：{len(self.frozen_zs)}")

    @staticmethod
    def _check_overlap(bis: List[BI]) -> bool:
        """检查候选笔序列是否有重叠（交集非空）"""
        return max(bi.low for bi in bis) <= min(bi.high for bi in bis)


def chan_third_bs_V240601(c: CZSC, **kwargs) -> OrderedDict:
    """
    基于A0中枢状态的三买卖点信号
    参数模板："{freq}_D{di}N{n}_三买卖V5"

    信号列表：
      - Signal('{freq}_D{di}N{n}_三买卖V5_多头_任意_任意_0')
      - Signal('{freq}_D{di}N{n}_三买卖V5_空头_任意_任意_0')
    """
    di = int(kwargs.get("di", 1))
    n = int(kwargs.get("n", 3))
    freq = c.freq.value
    k1, k2, k3 = f"{freq}_D{di}N{n}_三买卖V5".split('_')

    # 从缓存中获取A0State对象；若无则初始化
    state: A0State = c.cache.get('a0_state', A0State())
    latest_bi = get_sub_elements(c.bi_list, di=di, n=1)[0]
    state.update(latest_bi)
    c.cache['a0_state'] = state

    # 生成信号：仅当中枢结束且重叠区间有效时
    zg, zd = state.get_zs_range()
    if not (zg and zd and state.end_flag and state.trigger_bi):
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="持仓")

    print("\n*** [检查触发条件] ***")
    print(f"触发笔方向：{state.trigger_bi.direction}")
    print(f"中枢重叠区间: [ZG={zg:.2f}, ZD={zd:.2f}]")
    print(f"触发笔低点：{state.trigger_bi.low:.2f}, 触发笔高点：{state.trigger_bi.high:.2f}")

    # 三买信号：若触发笔为向下且其 low > 当前重叠区间上轨 (ZG)
    if state.trigger_bi.direction == "down" and state.trigger_bi.low > zg:
        print(f"📈 开多信号 @ 触发笔低点 {state.trigger_bi.low:.2f} > ZG {zg:.2f}")
        signal = create_single_signal(k1=k1, k2=k2, k3=k3, v1="开多")
        state._freeze_active_zs(state.trigger_bi)
        return signal

    # 三卖信号：若触发笔为向上且其 high < 当前重叠区间下轨 (ZD)
    if state.trigger_bi.direction == "up" and state.trigger_bi.high < zd:
        print(f"📉 开空信号 @ 触发笔高点 {state.trigger_bi.high:.2f} < ZD {zd:.2f}")
        signal = create_single_signal(k1=k1, k2=k2, k3=k3, v1="开空")
        state._freeze_active_zs(state.trigger_bi)
        return signal

    # 平仓信号示例：当冻结中枢数量>=2且最新笔突破上一个冻结中枢区间时触发平仓
    if len(state.frozen_zs) >= 2:
        last_zs = state.frozen_zs[-2]
        if (state.trigger_bi.direction == "up" and latest_bi.high > last_zs.zg) or \
           (state.trigger_bi.direction == "down" and latest_bi.low < last_zs.zd):
            print("🔄 平仓信号触发")
            return create_single_signal(k1=k1, k2=k2, k3=k3, v1="平仓")

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1="持仓")
