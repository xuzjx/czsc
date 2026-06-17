# -*- coding: utf-8 -*-
"""
author: 编码助手
describe: A0/A1 中枢突破策略 (严格对齐 MQL5 版本)

MQL5 -> CZSC 对应关系：
- UpdatePenState(h,l,o,c,t)        -> 信号函数内【核心逻辑 1：维护 A0 区间】
- Add_A0_To_Sequence(zone)         -> _add_a0_to_sequence
- EvaluateA1()                     -> _evaluate_a1
- ExecuteA1BreakoutTrading(close)  -> 信号函数内【核心逻辑 3：A1 突破开平仓】

重要设计意图（与 MQL5 一单止损版一致，刻意保留，不要"修复"）：
1. 旧 A1 不失效：突破开仓后只重置 A0 序列，has_a1 保持 True，旧 A1 继续作为
   中枢参考，直到 EvaluateA1 形成新 A1 才被覆盖。三买后价格跌回可基于同一个
   旧中枢出三卖反手，反手信号同时充当结构性止损。
2. A0 入列即定格：A0 序列里存的是脱离 K 线出生时的单根 K 线高低快照，
   之后 curr_zone 的压缩不回写数组里的副本。
3. 一单止损：不做金字塔加仓；开仓时按 In_sl * ATR 记录止损价，后续 K 线收盘
   触发止损平仓。czsc 回测无法使用 MT5 的 ASK/BID 即时价，这里用收盘价近似。
"""

from __future__ import annotations
from typing import List, Optional, OrderedDict

from czsc import CZSC, CzscStrategyBase, Operate, Signal, Factor, Event, Position
from czsc.utils import create_single_signal

from czsc.signals.strategies_pen_zone_mt5 import _log_mt5

import logging

logger = logging.getLogger(__name__)


def _zone_mid(z: dict) -> float:
    """区间中点（对齐 MQL5: (top + bottom) / 2，用于比较两个 A0 的方向）"""
    return (float(z["top"]) + float(z["bottom"])) / 2.0


def _new_zone(h: float, l: float, dt) -> dict:
    """以单根 K 线建立区间（对齐 MQL5 Zone 结构，start_time = end_time = t）"""
    return {"top": float(h), "bottom": float(l), "sdt": dt, "edt": dt}


def _calc_atr(c: CZSC, period: int = 14) -> float:
    """计算最近 period 根 K 线 ATR，近似 MT5 已收盘 K 线的 ATR 取值"""
    bars = c.bars_raw
    if len(bars) < 2:
        return float(bars[-1].high - bars[-1].low) if bars else 0.0

    trs = []
    start = max(1, len(bars) - max(1, int(period)))
    for i in range(start, len(bars)):
        bar = bars[i]
        prev_close = float(bars[i - 1].close)
        high, low = float(bar.high), float(bar.low)
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(trs) / len(trs) if trs else 0.0


def _evaluate_a1(c: CZSC, state: dict, log_on: bool, long_only: bool = False) -> Optional[str]:
    """对齐 MQL5: EvaluateA1()

    在每个新 A0 入列后调用，负责：
    - 维护 A0 序列的特殊情况 3 / 4（滚动数组）
    - 判断反向运动路径与 A0_01 是否重合 -> 形成 A1
    - A1【出生即突破】检查

    :return: "LONG" / "SHORT" 表示出生即突破触发的开仓方向；None 表示无交易
    """
    a0 = state["a0_array"]
    if len(a0) < 2:
        return None

    a0_01, a0_02 = a0[0], a0[1]
    # 方向用区间中点比较
    dir_01_02 = 1 if _zone_mid(a0_02) > _zone_mid(a0_01) else -1

    if len(a0) >= 3:
        latest_a0 = a0[-1]
        prev_a0 = a0[-2]
        dir_latest = 1 if _zone_mid(latest_a0) > _zone_mid(prev_a0) else -1

        # 特殊情况3：count==3 且与初始方向同向生成 -> 数组滚动为 [A0_02, latest]
        if len(a0) == 3 and dir_01_02 == dir_latest:
            state["a0_array"] = [a0_02, latest_a0]
            if log_on:
                _log_mt5(c, "A0_ROLL3", "count==3 same direction -> roll array to [A0_02, latest]",
                         dir_01_02=int(dir_01_02))
            return None

        # 寻找反向运动起点：从 i=count-2 往前找第一个 temp_dir == dir_01_02 的位置 i
        # （找不到则 i 默认 count-2）
        idx = len(a0) - 2
        for i in range(len(a0) - 2, 0, -1):
            temp_dir = 1 if _zone_mid(a0[i]) > _zone_mid(a0[i - 1]) else -1
            if temp_dir == dir_01_02:
                idx = i
                break
        reverse_start_a0 = a0[idx]

        # 反向运动路径区间，与 A0_01 判断是否重合
        path_top = max(reverse_start_a0["top"], latest_a0["top"])
        path_bottom = min(reverse_start_a0["bottom"], latest_a0["bottom"])
        is_overlap = path_bottom <= a0_01["top"] and path_top >= a0_01["bottom"]

        if is_overlap:
            # A1 形成：价格区间取 A0_01，end_time 延伸到 latest
            a1 = dict(a0_01)
            a1["edt"] = latest_a0["edt"]
            state["a1_zone"] = a1
            state["has_a1"] = True
            if log_on:
                _log_mt5(c, "A1_FORM", "reverse path overlaps A0_01 -> A1 formed",
                         a1_top=float(a1["top"]), a1_bottom=float(a1["bottom"]))

            # 【出生即突破】A1 形成瞬间检查（注意：数组重置后 has_a1 保持 True）
            if latest_a0["bottom"] > a1["top"] and state["pos_dir"] != 1:
                state["a0_array"] = [latest_a0]
                if log_on:
                    _log_mt5(c, "A1_BIRTH_LONG", "A1 born already broken upward -> reverse to long",
                             latest_bottom=float(latest_a0["bottom"]), a1_top=float(a1["top"]))
                return "LONG"
            if latest_a0["top"] < a1["bottom"] and state["pos_dir"] != -1:
                # ETF 只做多且空仓时，向下出生突破无平仓动作，不应重置 A0 序列
                if long_only and state["pos_dir"] != 1:
                    pass
                else:
                    state["a0_array"] = [latest_a0]
                    if log_on:
                        _log_mt5(c, "A1_BIRTH_SHORT", "A1 born already broken downward -> reverse to short",
                                 latest_top=float(latest_a0["top"]), a1_bottom=float(a1["bottom"]))
                    return "SHORT"
        else:
            # 特殊情况4：反向运动终止且未与 A0_01 重合 -> 数组滚动为 [prev, latest]
            if dir_latest == dir_01_02:
                state["a0_array"] = [prev_a0, latest_a0]
                if log_on:
                    _log_mt5(c, "A0_ROLL4", "reverse move ended without overlap -> roll array to [prev, latest]",
                             dir_latest=int(dir_latest))
    return None


def pen_zone_a1_signal_v1(c: CZSC, di: int = 1, **kwargs) -> OrderedDict:
    """A0/A1 中枢突破信号 (严格对齐 MQL5 版本)

    每根 K 线收盘时依次执行：
    1. UpdatePenState：K 线脱离当前区间则生成新 A0（并执行 EvaluateA1）；
       未脱离则压缩区间取交集
    2. ExecuteA1BreakoutTrading：收盘价突破 A1 -> 三买/三卖（平反向仓 + 开仓）

    信号列表：
    - Signal('15分钟_D1_PenZoneA1V1_平空开多_任意_任意_0')  三买（含 A1 出生即突破向上）
    - Signal('15分钟_D1_PenZoneA1V1_平多开空_任意_任意_0')  三卖（含 A1 出生即突破向下）
    - Signal('15分钟_D1_PenZoneA1V1_多头止损_任意_任意_0')  多单触发 ATR 止损
    - Signal('15分钟_D1_PenZoneA1V1_空头止损_任意_任意_0')  空单触发 ATR 止损

    :param c: CZSC 对象（signals_config 中指定 freq 后由框架传入；要求 freq == base_freq，
              保证每根 K 线只被处理一次且为已收盘 K 线）
    :param di: 固定为 1，只处理最后一根已收盘 K 线
    :param kwargs:
        - atr_period: ATR 计算周期，默认 14
        - in_sl: 止损 ATR 倍数，默认 6
        - use_atr_stop: 是否启用 ATR 止损，默认 True
        - use_a1_zone: 是否启用 A1 中枢过滤（反向路径与 A0_01 重合才形成 A1）；
          False 时跳过 EvaluateA1，有 2 个及以上 A0 即用 A0_01 作为中枢参考做突破交易
        - log: 是否输出对照 MQL5 Print 的执行日志
    """
    if di != 1:
        raise ValueError("This signal logic must run on the last closed bar (di=1)")

    k1 = str(getattr(c, "freq", kwargs.get("k1", "15分钟")))
    k2 = f"D{di}"
    long_only = bool(kwargs.get("long_only", False))
    k3 = "PenZoneA1EtfV1" if long_only else "PenZoneA1V1"

    if len(c.bars_raw) < 2:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    last_bar = c.bars_raw[-1]
    last_dt = last_bar.dt
    h, l, o, c0 = float(last_bar.high), float(last_bar.low), float(last_bar.open), float(last_bar.close)
    is_yang = c0 > o
    is_yin = c0 < o
    log_on = bool(kwargs.get("log", True))
    atr_period = int(kwargs.get("atr_period", 14))
    in_sl = float(kwargs.get("in_sl", 6))
    use_atr_stop = bool(kwargs.get("use_atr_stop", True))
    use_a1_zone = bool(kwargs.get("use_a1_zone", True))
    current_atr = _calc_atr(c, atr_period)

    # --- 时间闸门：同一根 K 线只处理一次（防止状态被重复推进）---
    if c.cache.get("pen_zone_a1_last_dt") == last_dt:
        return c.cache.get("pen_zone_a1_last_sig", create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他"))

    # --- 状态缓存 (对齐 MQL5 成员变量) ---
    state = c.cache.get("pen_zone_a1_state", {})
    if not state:
        state = {
            # m_curr_zone / m_prev_zone / m_zone_initialized
            "zone_initialized": False,
            "curr_zone": None,
            "prev_zone": None,
            # m_A0_array / m_A1_zone / m_has_A1
            "a0_array": [],
            "a1_zone": None,
            "has_a1": False,
            # 持仓方向跟踪（对齐 MQL5 HasPositionByType；反手时多空互斥，
            # 独立止损时内部状态会先归零，随后由 Position 平仓事件同步）
            "pos_dir": 0,       # 1 多 / -1 空 / 0 无
            "long_sl": None,    # 多单 ATR 止损价
            "short_sl": None,   # 空单 ATR 止损价
        }

    open_long, open_short, close_long = False, False, False
    stop_long, stop_short = False, False

    def _finish(v1: str) -> OrderedDict:
        c.cache["pen_zone_a1_state"] = state
        res = create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
        c.cache["pen_zone_a1_last_dt"] = last_dt
        c.cache["pen_zone_a1_last_sig"] = res
        return res

    # 十字星跳过 (对齐 MQL5: !isYang && !isYin -> return)
    if not (is_yang or is_yin):
        return _finish("其他")

    # ================= 核心逻辑 1: UpdatePenState（维护 A0 区间）=================
    if not state["zone_initialized"]:
        # 首次运行：第一根 K 线建立初始区间
        state["curr_zone"] = _new_zone(h, l, last_dt)
        state["zone_initialized"] = True
        if log_on:
            _log_mt5(c, "ZONE_INIT", "init first zone by bar", top=h, bottom=l)
    else:
        curr = state["curr_zone"]
        if l > curr["top"] or h < curr["bottom"]:
            # K 线整体脱离当前区间
            state["prev_zone"] = dict(curr)

            # --- 以当前脱离 K 线生成新 A0 ---
            state["curr_zone"] = _new_zone(h, l, last_dt)
            # Add_A0_To_Sequence：入列的是"出生快照"（单根 K 线高低），
            # 之后 curr_zone 的压缩不会回写数组里的副本（有意设计）
            state["a0_array"].append(dict(state["curr_zone"]))
            if log_on:
                _log_mt5(c, "A0_BORN", "bar detached zone -> new A0",
                         top=h, bottom=l, a0_count=len(state["a0_array"]))

            if use_a1_zone:
                action = _evaluate_a1(c, state, log_on, long_only=long_only)
                if action == "LONG":
                    open_long = True
                elif action == "SHORT":
                    if long_only:
                        if state["pos_dir"] == 1:
                            close_long = True
                            state["pos_dir"] = 0
                            state["long_sl"] = None
                    else:
                        open_short = True
            elif len(state["a0_array"]) >= 2:
                a0_01 = state["a0_array"][0]
                a1 = dict(a0_01)
                a1["edt"] = state["a0_array"][-1]["edt"]
                state["a1_zone"] = a1
                state["has_a1"] = True
        else:
            # 未脱离：区间压缩取交集
            curr["top"] = min(curr["top"], h)
            curr["bottom"] = max(curr["bottom"], l)
            curr["edt"] = last_dt
            if log_on:
                _log_mt5(c, "ZONE_COMPRESS", "bar inside zone -> compress",
                         top=float(curr["top"]), bottom=float(curr["bottom"]))

    # 出生即突破的开仓先落地持仓方向，使后面的 ExecuteA1BreakoutTrading 自然被"无多/空单"条件拦截
    if open_long:
        state["pos_dir"] = 1
        state["long_sl"] = c0 - in_sl * current_atr if (use_atr_stop and current_atr > 0) else None
        state["short_sl"] = None
    elif open_short:
        state["pos_dir"] = -1
        state["short_sl"] = c0 + in_sl * current_atr if (use_atr_stop and current_atr > 0) else None
        state["long_sl"] = None

    # ============ 核心逻辑 3: ExecuteA1BreakoutTrading（A1 突破开平仓）============
    if state["has_a1"] and state["a0_array"]:
        current_a0 = state["a0_array"][-1]  # 注意：数组里的出生快照
        a1 = state["a1_zone"]

        # 三买：current_A0.bottom > A1.top 且无多单 且 currentClose > A1.top
        if current_a0["bottom"] > a1["top"] and state["pos_dir"] != 1 and c0 > a1["top"]:
            open_long = True
            state["pos_dir"] = 1
            state["long_sl"] = c0 - in_sl * current_atr if (use_atr_stop and current_atr > 0) else None
            state["short_sl"] = None
            # 只重置 A0 序列；has_a1 保持 True，旧 A1 继续作为参考中枢
            state["a0_array"] = [current_a0]
            if log_on:
                _log_mt5(c, "A1_BREAK_LONG", "third buy: close shorts and open long",
                         close=c0, a1_top=float(a1["top"]), a0_bottom=float(current_a0["bottom"]))

        # 三卖：完全对称（ETF 只做多时仅平多；空仓时跳过，避免反复重置 A0 延迟下次开多）
        elif current_a0["top"] < a1["bottom"] and c0 < a1["bottom"]:
            if long_only:
                if state["pos_dir"] == 1:
                    state["a0_array"] = [current_a0]
                    close_long = True
                    state["pos_dir"] = 0
                    state["long_sl"] = None
                    if log_on:
                        _log_mt5(c, "A1_BREAK_CLOSE", "third sell -> close long (ETF long-only)",
                                 close=c0, a1_bottom=float(a1["bottom"]), a0_top=float(current_a0["top"]))
            elif state["pos_dir"] != -1:
                state["a0_array"] = [current_a0]
                open_short = True
                state["pos_dir"] = -1
                state["short_sl"] = c0 + in_sl * current_atr if (use_atr_stop and current_atr > 0) else None
                state["long_sl"] = None
                if log_on:
                    _log_mt5(c, "A1_BREAK_SHORT", "third sell: close longs and open short",
                             close=c0, a1_bottom=float(a1["bottom"]), a0_top=float(current_a0["top"]))

    # --- 一单 ATR 止损：反手信号优先，只有未反手时才输出独立止损 ---
    if use_atr_stop and not open_long and not open_short and not close_long:
        long_sl = state.get("long_sl")
        short_sl = state.get("short_sl")
        if state["pos_dir"] == 1 and long_sl is not None and c0 <= long_sl:
            stop_long = True
            state["pos_dir"] = 0
            state["long_sl"] = None
            if log_on:
                _log_mt5(c, "ATR_STOP_LONG", "close long by ATR stop", close=c0, sl=float(long_sl))
        elif not long_only and state["pos_dir"] == -1 and short_sl is not None and c0 >= short_sl:
            stop_short = True
            state["pos_dir"] = 0
            state["short_sl"] = None
            if log_on:
                _log_mt5(c, "ATR_STOP_SHORT", "close short by ATR stop", close=c0, sl=float(short_sl))

    # --- 信号解析 ---
    if long_only:
        if open_long:
            v1 = "开多"
        elif close_long:
            v1 = "平多"
        elif stop_long:
            v1 = "多头止损"
        else:
            v1 = "其他"
    elif open_long:
        v1 = "平空开多"
    elif open_short:
        v1 = "平多开空"
    elif stop_long:
        v1 = "多头止损"
    elif stop_short:
        v1 = "空头止损"
    else:
        v1 = "其他"

    return _finish(v1)


def get_events_pen_zone_a1(freq: str = "15分钟") -> List[Event]:
    """主力仓位事件：三买/三卖反手（开仓信号同时充当对侧的平仓信号，对齐 MQL5
    "平所有反向单 + 市价开仓"；另有一单 ATR 止损信号）"""
    s_rev_long = Signal(f"{freq}_D1_PenZoneA1V1_平空开多_任意_任意_0")
    s_rev_short = Signal(f"{freq}_D1_PenZoneA1V1_平多开空_任意_任意_0")
    s_long_sl = Signal(f"{freq}_D1_PenZoneA1V1_多头止损_任意_任意_0")
    s_short_sl = Signal(f"{freq}_D1_PenZoneA1V1_空头止损_任意_任意_0")

    return [
        Event(name="开多", operate=Operate.LO, factors=[Factor(name="F1", signals_all=[s_rev_long])]),
        Event(name="开空", operate=Operate.SO, factors=[Factor(name="F2", signals_all=[s_rev_short])]),
        Event(name="平多", operate=Operate.LE, factors=[Factor(name="F3", signals_all=[s_rev_short])]),
        Event(name="平空", operate=Operate.SE, factors=[Factor(name="F4", signals_all=[s_rev_long])]),
        Event(name="多头止损", operate=Operate.LE, factors=[Factor(name="F5", signals_all=[s_long_sl])]),
        Event(name="空头止损", operate=Operate.SE, factors=[Factor(name="F6", signals_all=[s_short_sl])]),
    ]


class PenZoneA1Strategy(CzscStrategyBase):
    """A0/A1 中枢突破策略：主力仓位（三买/三卖反手）+ 一单 ATR 止损"""

    base_freq = "15分钟"

    @property
    def positions(self) -> List[Position]:
        e = get_events_pen_zone_a1(self.base_freq)
        return [
            Position(symbol=self.symbol, name="PenZoneA1V1", opens=[e[0], e[1]], exits=e[2:], T0=True),
        ]

    @property
    def signals_config(self):
        return [
            # 指定 freq，让框架传入对应周期的 CZSC 对象；freq 必须等于 base_freq，
            # 保证信号函数每根 K 线只在收盘后被推进一次
            {"name": pen_zone_a1_signal_v1, "freq": self.base_freq, "di": 1, "log": True,
             "atr_period": 14, "in_sl": 6},
        ]


def get_events_pen_zone_a1_etf(freq: str = "30分钟") -> List[Event]:
    """A股 ETF 只做多 T+1：三买开多，三卖/止损平多（不可做空、不可当日卖出）"""
    s_open = Signal(f"{freq}_D1_PenZoneA1EtfV1_开多_任意_任意_0")
    s_close = Signal(f"{freq}_D1_PenZoneA1EtfV1_平多_任意_任意_0")
    s_long_sl = Signal(f"{freq}_D1_PenZoneA1EtfV1_多头止损_任意_任意_0")

    return [
        Event(name="开多", operate=Operate.LO, factors=[Factor(name="F1", signals_all=[s_open])]),
        Event(name="平多", operate=Operate.LE, factors=[Factor(name="F2", signals_all=[s_close])]),
        Event(name="多头止损", operate=Operate.LE, factors=[Factor(name="F3", signals_all=[s_long_sl])]),
    ]


class PenZoneA1EtfStrategy(CzscStrategyBase):
    """A股 ETF 只做多 T+1 版 PenZoneA1：三买开多，三卖/ATR 止损平多"""

    base_freq = "30分钟"

    @property
    def positions(self) -> List[Position]:
        e = get_events_pen_zone_a1_etf(self.base_freq)
        return [
            Position(symbol=self.symbol, name="PenZoneA1EtfV1", opens=[e[0]], exits=e[1:], T0=False),
        ]

    @property
    def signals_config(self):
        return [
            {"name": pen_zone_a1_signal_v1, "freq": self.base_freq, "di": 1, "log": True,
             "atr_period": 14, "in_sl": 6, "long_only": True},
        ]
