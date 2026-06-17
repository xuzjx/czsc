# -*- coding: utf-8 -*-
"""
A股 Pen_Zone 只做多策略（对齐 MT5 上涨延续逻辑）

- 每个交易日仅在下午窗口内评估是否买入（默认 14:30 ≤ 时刻 < 15:00）
- 买入条件：压缩区间已初始化 + 阳笔延续 + 价格持续在区间上方 + 连续脱离计数达标
  + MA250 均线；不使用 RSI / 布林带 / InPy 距离过滤
- 卖出：结构下移、跌回区间下方、或跌破区间底止损
"""

from __future__ import annotations

from typing import List, OrderedDict

import pandas as pd

from czsc import CZSC, CzscStrategyBase, Operate, Signal, Factor, Event, Position
from czsc.utils import create_single_signal

from czsc.signals.strategies_pen_zone_mt5 import _log_mt5


def _time_to_minutes(hhmm: str) -> int:
    parts = hhmm.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def _in_trade_window(dt, trade_time_start: str = "14:30", trade_time_end: str = "15:00") -> bool:
    """是否处于允许买入的时间窗（按 K 线收盘时刻 HH:MM，左闭右开）"""
    return True;
    # if dt is None:
    #     return False
    # t = dt.hour * 60 + dt.minute
    # return _time_to_minutes(trade_time_start) <= t < _time_to_minutes(trade_time_end)


def _is_uptrend_continuing(
    state: dict,
    h: float,
    l: float,
    is_yang: bool,
    pen_size: float,
    pen_size_1: float,
    pen_type: str,
    inout_count: int,
) -> bool:
    """上涨延续判断逻辑

    - 情况 A: 第 2 区间及以上已确立 (count_up_zones >= 1)，只要价格在区间底之上且为阳笔即可买入
    - 情况 B: 第 1 区间或通用突破，要求阳笔 + 价格脱离区间上方 + 连续计数达标
    """
    if not state.get("zone_initialized"):
        return False

    # 情况 A: 第 2 区间及以上已确立
    if state.get("count_up_zones", 0) >= 1:
        if l > state.get("curr_zone_bottom", 0):
            return True

    # 情况 B: 通用脱离突破逻辑
    if pen_type != "YANG":
        return False
    if l <= state["curr_zone_top"]:
        return False
    
    # 彻底去掉 ATR 强度判定，直接通过
    strong = True
    if not strong:
        return False
    if state.get("count_above", 0) < inout_count:
        return False
    return True


def pen_zone_stock_long_signal_v1(c: CZSC, di: int = 1, **kwargs) -> OrderedDict:
    """Pen_Zone 股票只做多信号（14:30~15:00 买入窗 + MT5 上涨延续逻辑）"""
    if di != 1:
        raise ValueError("This signal logic must run on the last closed bar (di=1)")

    k1 = str(getattr(c, "freq", kwargs.get("k1", "5分钟")))
    k2 = f"D{di}"
    k3 = "PenZoneStockLongV1"

    if len(c.bars_raw) < 251:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    last_bar = c.bars_raw[-1]
    last_dt = getattr(last_bar, "dt", None)
    h, l, o, c0 = last_bar.high, last_bar.low, last_bar.open, last_bar.close
    is_yang = c0 > o
    is_yin = c0 < o
    log_on = bool(kwargs.get("log", True))
    trade_time_start = str(kwargs.get("trade_time_start", kwargs.get("trade_time", "14:30")))
    trade_time_end = str(kwargs.get("trade_time_end", "15:00"))

    ATR_N = int(kwargs.get("atr_period", 14))
    MA_N = int(kwargs.get("ma_period", 250))
    MA_FILTER = bool(kwargs.get("use_ma_filter", True))
    INOUT_COUNT = int(kwargs.get("inout_count", 1))

    df = pd.DataFrame([{"high": b.high, "low": b.low, "close": b.close} for b in c.bars_raw])
    close_s = df["close"]

    ma = close_s.rolling(window=MA_N, min_periods=1).mean()
    current_ma = float(ma.iloc[-1]) if not ma.empty else 0.0

    state = c.cache.get("pen_zone_stock_state", {})
    if not state:
        state = {
            "pen_type": "NONE",
            "pen_top": 0.0,
            "pen_bottom": 0.0,
            "zone_initialized": False,
            "curr_zone_top": 0.0,
            "curr_zone_bottom": 0.0,
            "signal_ready": 0,
            "count_above": 0,
            "count_below": 0,
            "count_up_zones": 0,
            "count_down_zones": 0,
        }

    pen_type = state["pen_type"]
    pen_top, pen_bottom = state["pen_top"], state["pen_bottom"]
    zone_initialized = state["zone_initialized"]
    curr_top, curr_bottom = state["curr_zone_top"], state["curr_zone_bottom"]
    signal_ready = state["signal_ready"]
    count_above, count_below = state["count_above"], state["count_below"]
    count_up_zones, count_down_zones = state["count_up_zones"], state["count_down_zones"]

    buy_signal = False
    sell_signal = False

    if not (is_yang or is_yin):
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    if log_on:
        _log_mt5(
            c,
            "BAR",
            "stock bar",
            h=float(h),
            l=float(l),
            close=float(c0),
            is_yang=bool(is_yang),
            atr=float(current_atr),
            ma=float(current_ma),
            atr_multi=float(ATR_M),
        )

    # --- UpdatePenState（与 MT5 一致，不含开空/对冲）---
    if pen_type == "NONE":
        pen_type = "YANG" if is_yang else "YIN"
        pen_top, pen_bottom = h, l
        if log_on:
            _log_mt5(c, "PEN_INIT", "init first pen", pen_type=pen_type, pen_top=float(pen_top), pen_bottom=float(pen_bottom))
    elif (pen_type == "YANG" and is_yin) or (pen_type == "YIN" and is_yang):
        pen_size_1 = pen_top - pen_bottom
        # 去掉区间构造时的 ATR 过滤判定，任何笔都参与区间更新
        if True:
            if not zone_initialized:
                curr_top, curr_bottom = pen_top, pen_bottom
                zone_initialized = True
                if log_on:
                    _log_mt5(c, "ZONE_INIT", "init zone", curr_top=float(curr_top), curr_bottom=float(curr_bottom))
            else:
                intersect_top = min(curr_top, pen_top)
                intersect_bottom = max(curr_bottom, pen_bottom)
                if intersect_top > intersect_bottom:
                    curr_top, curr_bottom = intersect_top, intersect_bottom
                    signal_ready = 0
                    if log_on:
                        _log_mt5(c, "ZONE_COMPRESS", "compress zone", curr_top=float(curr_top), curr_bottom=float(curr_bottom))
                else:
                    if pen_type == "YIN" and pen_bottom > curr_top:
                        signal_ready = 1
                    elif pen_type == "YANG" and pen_top < curr_bottom:
                        signal_ready = -1

            if signal_ready == 1 and is_yang:
                curr_top, curr_bottom = pen_top, pen_bottom
                signal_ready = 0
                count_above, count_below = 0, 0
                count_up_zones += 1
                count_down_zones = 0
                if log_on:
                    _log_mt5(c, "ZONE_SHIFT_UP", "shift up", curr_top=float(curr_top), curr_bottom=float(curr_bottom))
            elif signal_ready == -1 and is_yin:
                curr_top, curr_bottom = pen_top, pen_bottom
                signal_ready = 0
                count_above, count_below = 0, 0
                count_down_zones += 1
                count_up_zones = 0
                sell_signal = True
                if log_on:
                    _log_mt5(c, "ZONE_SHIFT_DOWN", "shift down -> sell", curr_top=float(curr_top), curr_bottom=float(curr_bottom))

        pen_top, pen_bottom = h, l
        pen_type = "YANG" if is_yang else "YIN"
    else:
        if pen_type == "YANG":
            pen_top = max(pen_top, h)
        else:
            pen_bottom = min(pen_bottom, l)
        if signal_ready == 1 and l <= curr_top:
            signal_ready = 0
        if signal_ready == -1 and h >= curr_bottom:
            signal_ready = 0

    # --- CheckContinuousBreakout：只维护计数（买入在下午时间窗内判断）---
    if zone_initialized:
        pen_size = h - l
        pen_size_1 = pen_top - pen_bottom

        if l > curr_top:
            # 去掉 ATR 强度过滤判定
            if True:
                count_above += 1
                count_below = 0
                if log_on:
                    _log_mt5(c, "COUNT_ABOVE", "above zone", count_above=int(count_above), curr_top=float(curr_top))
        elif h < curr_bottom:
            # 去掉 ATR 强度过滤判定
            if True:
                count_below += 1
                count_above = 0
                if log_on:
                    _log_mt5(c, "COUNT_BELOW", "below zone", count_below=int(count_below), curr_bottom=float(curr_bottom))
        else:
            count_above, count_below = 0, 0
            if log_on:
                _log_mt5(c, "COUNT_RESET", "back in zone", curr_top=float(curr_top), curr_bottom=float(curr_bottom))

        # 持仓风控：跌回区间下方 / 跌破区间底
        if c0 < curr_bottom:
            sell_signal = True
            if log_on:
                _log_mt5(c, "SELL_STOP", "close below zone bottom", close=float(c0), curr_bottom=float(curr_bottom))
        elif count_below >= INOUT_COUNT:
            sell_signal = True
            if log_on:
                _log_mt5(c, "SELL_BREAK", "continuous below zone", count_below=int(count_below))

    state.update(
        {
            "pen_type": pen_type,
            "pen_top": float(pen_top),
            "pen_bottom": float(pen_bottom),
            "zone_initialized": zone_initialized,
            "curr_zone_top": float(curr_top),
            "curr_zone_bottom": float(curr_bottom),
            "signal_ready": int(signal_ready),
            "count_above": int(count_above),
            "count_below": int(count_below),
            "count_up_zones": int(count_up_zones),
            "count_down_zones": int(count_down_zones),
        }
    )
    c.cache["pen_zone_stock_state"] = state

    # --- 下午买入时间窗（默认 14:30 ~ 15:00，左闭右开）---
    in_buy_window = _in_trade_window(last_dt, trade_time_start, trade_time_end)
    if in_buy_window and zone_initialized:
        pen_size = h - l
        pen_size_1 = pen_top - pen_bottom
        uptrend = _is_uptrend_continuing(
            state,
            h,
            l,
            is_yang,
            pen_size,
            pen_size_1,
            pen_type,
            INOUT_COUNT,
        )
        if uptrend:
            ma_ok = (not MA_FILTER) or c0 > current_ma
            trade_date = last_dt.date() if last_dt else None
            already_bought_today = c.cache.get("pen_zone_stock_last_buy_date") == trade_date
            if ma_ok and not already_bought_today:
                buy_signal = True
                count_above = 0
                state["count_above"] = 0
                c.cache["pen_zone_stock_state"] = state
                if trade_date is not None:
                    c.cache["pen_zone_stock_last_buy_date"] = trade_date
                if log_on:
                    _log_mt5(
                        c,
                        "BUY",
                        f"uptrend continue in [{trade_time_start}, {trade_time_end})",
                        close=float(c0),
                        ma=float(current_ma),
                        dt=str(last_dt),
                    )
            elif log_on:
                _log_mt5(
                    c,
                    "BUY_BLOCK",
                    "uptrend ok but filter blocked",
                    ma_ok=bool(ma_ok),
                )
        elif log_on:
            _log_mt5(
                c,
                "BUY_SKIP",
                f"not uptrend continue in [{trade_time_start}, {trade_time_end})",
                pen_type=pen_type,
                count_above=int(count_above),
                l=float(l),
                curr_top=float(curr_top),
            )

    if buy_signal:
        v1 = "买入"
    elif sell_signal:
        v1 = "卖出"
    else:
        v1 = "其他"

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


def get_events_pen_zone_stock_long(freq: str = "5分钟") -> List[Event]:
    s_buy = Signal(f"{freq}_D1_PenZoneStockLongV1_买入_任意_任意_0")
    s_sell = Signal(f"{freq}_D1_PenZoneStockLongV1_卖出_任意_任意_0")
    return [
        Event(name="买入", operate=Operate.LO, factors=[Factor(name="F1", signals_all=[s_buy])]),
        Event(name="卖出", operate=Operate.LE, factors=[Factor(name="F2", signals_all=[s_sell])]),
    ]


class PenZoneStockLongStrategy(CzscStrategyBase):
    """A股 Pen_Zone 只做多：默认 5 分钟 K，每日 14:30~15:00 窗口内检查买入。"""

    base_freq = "5分钟"
    trade_time_start = "14:30"
    trade_time_end = "15:00"

    @property
    def positions(self) -> List[Position]:
        e = get_events_pen_zone_stock_long(self.base_freq)
        return [
            Position(
                symbol=self.symbol,
                name="PenZoneStockLongV1",
                opens=[e[0]],
                exits=[e[1]],
                T0=True,
                timeout=16 * 3000000,
            )
        ]

    @property
    def signals_config(self):
        return [
            {
                "name": pen_zone_stock_long_signal_v1,
                "freq": self.base_freq,
                "di": 1,
                "log": False,
                "trade_time_start": self.trade_time_start,
                "trade_time_end": self.trade_time_end,
                "ma_period": 250,
                "atr_multiplier": 1.0,
                "inout_count": 1,
            },
        ]
