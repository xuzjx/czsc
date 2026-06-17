# -*- coding: utf-8 -*-
"""
author: 编码助手
describe: 严格对齐用户贴出的 MQ5: Pen_Zone_Single_System.mq5

核心特征（以用户贴的 MQ5 为准）：
- 笔：用“K线颜色(阳/阴)延续、转色即结束笔”来维护 m_pen_top / m_pen_bottom
- 区间：用笔与区间的交集做“压缩区间”；无交集则产生预备信号，待“笔转色”确认区间切换
- 连续脱离计数：只要 K 线整体在区间上方 / 下方就计数（用户代码中 ATR / 笔长度过滤被注释掉）
- 入场：count>=Inout_count 触发，先平反向（本实现已去掉 InPy 距离过滤，达到计数即入场）
"""

from __future__ import annotations

from typing import List, OrderedDict

import numpy as np
import pandas as pd

from czsc import CZSC, CzscStrategyBase, Operate, Signal, Factor, Event, Position
from czsc.utils import create_single_signal
from czsc.signals.strategies_pen_zone_mt5 import _log_mt5, _get_dynamic_atr_multiplier


def pen_zone_single_system_signal_mt5_v1(c: CZSC, di: int = 1, **kwargs) -> OrderedDict:
    """主力信号：严格对齐用户贴的 MQ5 版本（过滤器关闭、脱离计数无条件累加）"""
    if di != 1:
        raise ValueError("This signal logic must run on the last closed bar (di=1)")

    k1 = str(getattr(c, "freq", kwargs.get("k1", "15分钟")))
    k2 = f"D{di}"
    k3 = "PenZoneSingleSystemMt5V1"

    if len(c.bars_raw) < 50:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    last_bar = c.bars_raw[-1]
    h, l, o, close = float(last_bar.high), float(last_bar.low), float(last_bar.open), float(last_bar.close)
    is_yang = close > o
    is_yin = close < o
    if not (is_yang or is_yin):
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    log_on = bool(kwargs.get("log", True))

    # === 参数（按 MQ5 变量名对齐；但过滤器在用户代码中基本关闭，这里保留参数便于复现实盘） ===
    ATR_N = int(kwargs.get("atr_period", 14))
    # 兼容保留参数：原 MQ5 用 IN_PY 做“距离过滤”；这里按用户要求去掉该限制
    IN_PY = float(kwargs.get("in_py", 10))
    INOUT_COUNT = int(kwargs.get("inout_count", 1))

    BB_N = int(kwargs.get("bb_period", 20))
    BB_K = float(kwargs.get("bb_dev", 2.0))
    BASE_ATR_M = float(kwargs.get("base_atr_multiplier", 1.0))
    MAX_ATR_M = float(kwargs.get("max_atr_multiplier", 4.0))
    IN_PERIOD = int(kwargs.get("in_period", 170))

    # --- 指标：动态 ATR 倍数用于日志对照（用户主逻辑中未使用过滤） ---
    df = pd.DataFrame([{"high": b.high, "low": b.low, "close": b.close} for b in c.bars_raw])
    high_s, low_s, close_s = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high_s - low_s, (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=ATR_N, min_periods=1).mean()
    current_atr = float(atr.iloc[-1]) if len(atr) else 0.0

    dynamic_atr_multi = _get_dynamic_atr_multiplier(
        close_s,
        bb_period=BB_N,
        bb_dev=BB_K,
        period=IN_PERIOD,
        base_multiplier=BASE_ATR_M,
        max_multiplier=MAX_ATR_M,
    )

    # === 状态（严格对齐 MQ5 成员变量） ===
    state = c.cache.get("pen_zone_single_system_mt5_state", {})
    if not state:
        state = {
            "pen_type": "NONE",  # NONE / YANG / YIN
            "pen_top": 0.0,
            "pen_bottom": 0.0,
            "zone_initialized": False,
            "curr_zone_top": 0.0,
            "curr_zone_bottom": 0.0,
            "signal_ready": 0,  # 1 预备做多；-1 预备做空；0 无
            "count_above": 0,
            "count_below": 0,
            "count_up_zones": 0,
            "count_down_zones": 0,
        }

    pen_type = state["pen_type"]
    pen_top, pen_bottom = float(state["pen_top"]), float(state["pen_bottom"])
    zone_initialized = bool(state["zone_initialized"])
    curr_top, curr_bottom = float(state["curr_zone_top"]), float(state["curr_zone_bottom"])
    signal_ready = int(state["signal_ready"])
    count_above, count_below = int(state["count_above"]), int(state["count_below"])
    count_up_zones, count_down_zones = int(state["count_up_zones"]), int(state["count_down_zones"])

    open_long, open_short = False, False
    exit_long, exit_short = False, False

    if log_on:
        _log_mt5(
            c,
            "BAR",
            "single_system bar inputs",
            h=h,
            l=l,
            o=o,
            close=close,
            is_yang=bool(is_yang),
            is_yin=bool(is_yin),
            atr=float(current_atr),
            dynamic_atr_multi=float(dynamic_atr_multi),
            curr_top=float(curr_top),
            curr_bottom=float(curr_bottom),
            signal_ready=int(signal_ready),
            count_above=int(count_above),
            count_below=int(count_below),
        )

    # =========================================================
    # 逻辑 1：UpdatePenState + OnPenCompleted（对齐用户贴的 MQ5）
    # =========================================================
    if pen_type == "NONE":
        pen_type = "YANG" if is_yang else "YIN"
        pen_top, pen_bottom = h, l
        if log_on:
            _log_mt5(c, "PEN_INIT", "init first pen", pen_type=pen_type, pen_top=pen_top, pen_bottom=pen_bottom)
    elif (pen_type == "YANG" and is_yin) or (pen_type == "YIN" and is_yang):
        # 用户贴的 MQ5 中：笔长度过滤被注释掉，这里不做过滤

        # 首次运行：建立初始区间（用刚完成的旧笔）
        if not zone_initialized:
            curr_top, curr_bottom = pen_top, pen_bottom
            zone_initialized = True
            if log_on:
                _log_mt5(c, "ZONE_INIT", "init zone by first completed pen", curr_top=curr_top, curr_bottom=curr_bottom)

        # OnPenCompleted：用刚完成的旧笔和当前区间求交集
        intersect_top = min(curr_top, pen_top)
        intersect_bottom = max(curr_bottom, pen_bottom)
        if intersect_top > intersect_bottom:
            curr_top, curr_bottom = intersect_top, intersect_bottom
            signal_ready = 0
            if log_on:
                _log_mt5(c, "ZONE_COMPRESS", "zone intersection exists -> compress", curr_top=curr_top, curr_bottom=curr_bottom)
        else:
            # 无交集：设置预备信号
            if pen_type == "YIN" and pen_bottom > curr_top:
                signal_ready = 1
                if log_on:
                    _log_mt5(c, "PREP_LONG", "yin pen above zone -> wait yang", pen_bottom=pen_bottom, curr_top=curr_top)
            elif pen_type == "YANG" and pen_top < curr_bottom:
                signal_ready = -1
                if log_on:
                    _log_mt5(c, "PREP_SHORT", "yang pen below zone -> wait yin", pen_top=pen_top, curr_bottom=curr_bottom)

        # 预备信号确认：等到“新颜色 K线”出现后，区间切换 & 平反向（MQ5：ClosePositionsByType）
        if signal_ready == 1 and is_yang:
            curr_top, curr_bottom = pen_top, pen_bottom
            signal_ready = 0
            count_above, count_below = 0, 0
            count_up_zones += 1
            count_down_zones = 0
            exit_short = True
            if log_on:
                _log_mt5(c, "ZONE_SHIFT_UP", "prep long confirmed -> shift zone", curr_top=curr_top, curr_bottom=curr_bottom, count_up_zones=count_up_zones)
        elif signal_ready == -1 and is_yin:
            curr_top, curr_bottom = pen_top, pen_bottom
            signal_ready = 0
            count_above, count_below = 0, 0
            count_down_zones += 1
            count_up_zones = 0
            exit_long = True
            if log_on:
                _log_mt5(c, "ZONE_SHIFT_DOWN", "prep short confirmed -> shift zone", curr_top=curr_top, curr_bottom=curr_bottom, count_down_zones=count_down_zones)

        # 开启新笔（新颜色笔从当前 K 开始）
        pen_type = "YANG" if is_yang else "YIN"
        pen_top, pen_bottom = h, l
        if log_on:
            _log_mt5(c, "PEN_SWITCH", "pen color reversed -> start new pen", pen_type=pen_type, pen_top=pen_top, pen_bottom=pen_bottom)
    else:
        # 笔延续：更新 top/bottom
        if pen_type == "YANG":
            pen_top = max(pen_top, h)
        else:
            pen_bottom = min(pen_bottom, l)

        # 失效检查：预备信号出现后，价格回区间则取消（对齐 MQ5）
        if signal_ready == 1 and l <= curr_top:
            signal_ready = 0
        if signal_ready == -1 and h >= curr_bottom:
            signal_ready = 0

    # =========================================================
    # 逻辑 2：CheckContinuousBreakout（对齐用户贴的 MQ5 版本）
    # =========================================================
    if zone_initialized:
        if l > curr_top:
            # 用户贴的 MQ5：计数逻辑无条件累加（原过滤被注释）
            count_above += 1
            count_below = 0
            if log_on:
                _log_mt5(c, "COUNT_ABOVE", "above zone counted (no filter)", count_above=count_above, curr_top=curr_top)
        elif h < curr_bottom:
            count_below += 1
            count_above = 0
            if log_on:
                _log_mt5(c, "COUNT_BELOW", "below zone counted (no filter)", count_below=count_below, curr_bottom=curr_bottom)
        else:
            count_above, count_below = 0, 0

        # --- 多：连续脱离触发，先平空，再开多（去掉 IN_PY 距离限制） ---
        if count_above >= INOUT_COUNT:
            exit_short = True
            aa = float(np.round(close - curr_top, 10))
            open_long = True
            count_above = 0
            if log_on:
                _log_mt5(
                    c,
                    "OPEN_LONG",
                    "open long by breakout (no in_py filter)",
                    aa=aa,
                    count_up_zones=count_up_zones,
                    in_py=float(IN_PY),
                )

        # --- 空：连续脱离触发，先平多，再开空（去掉 IN_PY 距离限制） ---
        if count_below >= INOUT_COUNT:
            exit_long = True
            aa = float(np.round(curr_bottom - close, 10))
            open_short = True
            count_below = 0
            if log_on:
                _log_mt5(
                    c,
                    "OPEN_SHORT",
                    "open short by breakout (no in_py filter)",
                    aa=aa,
                    count_down_zones=count_down_zones,
                    in_py=float(IN_PY),
                )

    # --- 回写缓存 ---
    state.update(
        {
            "pen_type": pen_type,
            "pen_top": float(pen_top),
            "pen_bottom": float(pen_bottom),
            "zone_initialized": bool(zone_initialized),
            "curr_zone_top": float(curr_top),
            "curr_zone_bottom": float(curr_bottom),
            "signal_ready": int(signal_ready),
            "count_above": int(count_above),
            "count_below": int(count_below),
            "count_up_zones": int(count_up_zones),
            "count_down_zones": int(count_down_zones),
        }
    )
    c.cache["pen_zone_single_system_mt5_state"] = state

    # --- 输出 v1 ---
    if open_long and exit_short:
        v1 = "平空开多"
    elif open_short and exit_long:
        v1 = "平多开空"
    elif open_long:
        v1 = "开多"
    elif open_short:
        v1 = "开空"
    elif exit_long:
        v1 = "平多"
    elif exit_short:
        v1 = "平空"
    else:
        v1 = "其他"

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


def get_events_pen_zone_single_system_mt5(freq: str = "15分钟") -> List[Event]:
    """主力仓位事件映射"""
    s_rev_long = Signal(f"{freq}_D1_PenZoneSingleSystemMt5V1_平空开多_任意_任意_0")
    s_rev_short = Signal(f"{freq}_D1_PenZoneSingleSystemMt5V1_平多开空_任意_任意_0")
    s_open_long = Signal(f"{freq}_D1_PenZoneSingleSystemMt5V1_开多_任意_任意_0")
    s_open_short = Signal(f"{freq}_D1_PenZoneSingleSystemMt5V1_开空_任意_任意_0")
    s_exit_long = Signal(f"{freq}_D1_PenZoneSingleSystemMt5V1_平多_任意_任意_0")
    s_exit_short = Signal(f"{freq}_D1_PenZoneSingleSystemMt5V1_平空_任意_任意_0")

    return [
        Event(name="开多", operate=Operate.LO, factors=[Factor(name="F1", signals_all=[s_open_long]), Factor(name="F1R", signals_all=[s_rev_long])]),
        Event(name="开空", operate=Operate.SO, factors=[Factor(name="F2", signals_all=[s_open_short]), Factor(name="F2R", signals_all=[s_rev_short])]),
        Event(name="平多", operate=Operate.LE, factors=[Factor(name="F3", signals_all=[s_exit_long, s_open_short, s_rev_short])]),
        Event(name="平空", operate=Operate.SE, factors=[Factor(name="F4", signals_all=[s_exit_short, s_open_long, s_rev_long])]),
    ]


class PenZoneSingleSystemMt5Strategy(CzscStrategyBase):
    """MQ5 Pen_Zone_Single_System 的 Python/CZSC 对齐实现"""

    base_freq = "15分钟"

    @property
    def positions(self) -> List[Position]:
        e = get_events_pen_zone_single_system_mt5(self.base_freq)
        return [
            Position(symbol=self.symbol, name="PenZoneSingleSystemMt5V1", opens=[e[0], e[1]], exits=[e[2], e[3]], T0=True),
        ]

    @property
    def signals_config(self):
        return [
            {"name": pen_zone_single_system_signal_mt5_v1, "freq": self.base_freq, "di": 1, "log": True},
        ]

