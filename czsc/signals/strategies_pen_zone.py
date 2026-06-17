# -*- coding: utf-8 -*-
"""
author: 编码助手
describe: 笔压缩区间 + 3分钟实时触价策略 (最终优化版)
新增：同步 MT5 的失效检查逻辑（回踩区间取消预备信号）
"""

from __future__ import annotations
from typing import List, OrderedDict
import pandas as pd
import numpy as np
import logging

from czsc import CZSC, CzscStrategyBase, Operate, Signal, Factor, Event, Position
from czsc.utils import create_single_signal

logger = logging.getLogger(__name__)

# --- 【全局状态桥梁】 ---
_STATE_BRIDGE = {}


def pen_zone_signal_V1(c: CZSC, di: int = 1, **kwargs) -> OrderedDict:
    """Pen_Zone 笔压缩区间突破信号 (严格对齐 MQL5 版本)"""
    # Since this function uses a global state bridge, it doesn't need to be a CzscSignals method.
    # The logic remains based on the CZSC object and the shared _STATE_BRIDGE.

    freq = c.freq.value
    symbol = c.symbol
    k1, k2, k3 = freq, f"D{di}", "PenZoneV1"

    if len(c.bars_raw) < 251:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    # --- 1. 时间闸门 ---
    last_bar = c.bars_raw[-1]
    last_dt = last_bar.dt
    if c.cache.get("last_processed_dt") == last_dt:
        return c.cache.get("last_v1_signal", create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他"))

    h, l, o, c0 = last_bar.high, last_bar.low, last_bar.open, last_bar.close
    is_yang, is_yin = c0 > o, c0 < o

    # --- 2. 参数指标 ---
    ATR_N = int(kwargs.get("atr_period", 14))
    ATR_M = float(kwargs.get("atr_multiplier", 1.0))
    MA_N = int(kwargs.get("ma_period", 250))
    RSI_U, RSI_L = float(kwargs.get("rsi_upper", 55)), float(kwargs.get("rsi_lower", 45))
    IN_PY = ATR_N * 3

    # 只取末尾部分进行转换
    relevant_bars = c.bars_raw[-MA_N:]

    close_arr = np.array([b.close for b in relevant_bars])
    high_arr = np.array([b.high for b in relevant_bars])
    low_arr = np.array([b.low for b in relevant_bars])

    tr = np.maximum(high_arr[1:] - low_arr[1:],
                    np.maximum(np.abs(high_arr[1:] - close_arr[:-1]), np.abs(low_arr[1:] - close_arr[:-1])))
    current_atr = np.mean(tr[-ATR_N:])

    delta = np.diff(close_arr)
    gain, loss = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    avg_loss = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    current_rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))

    current_ma = np.mean(close_arr[-MA_N:])
    bb_mid = pd.Series(close_arr).rolling(window=20).mean()
    bb_std = pd.Series(close_arr).rolling(window=20).std()
    is_expanding = (bb_mid + 2 * bb_std).diff().iloc[-1] > 0

    # --- 3. 状态维护 ---
    state = c.cache.get("pen_zone_state", {
        "pen_type": "NONE", "pen_top": 0.0, "pen_bottom": 0.0,
        "zone_initialized": False, "curr_zone_top": 0.0, "curr_zone_bottom": 0.0,
        "prev_zone_top": 0.0, "prev_zone_bottom": 0.0,
        "signal_ready": 0, "count_above": 0, "count_below": 0,
        "count_up_zones": 0, "count_down_zones": 0
    })

    v1 = "其他"
    exit_reason = ""  # 记录离场原因

    # --- 4. 笔与区间核心逻辑 ---

    if not (is_yang or is_yin):
        pass
    elif state["pen_type"] == "NONE":
        state["pen_type"], state["pen_top"], state["pen_bottom"] = ("YANG" if is_yang else "YIN"), h, l
    elif (state["pen_type"] == "YANG" and is_yin) or (state["pen_type"] == "YIN" and is_yang):
        pen_size = state["pen_top"] - state["pen_bottom"]

        if pen_size > current_atr * 2 * ATR_M:
            if not state["zone_initialized"]:
                state["curr_zone_top"], state["curr_zone_bottom"] = state["pen_top"], state["pen_bottom"]
                state["zone_initialized"] = True

                print(f"[{last_dt}] | 区间监控： atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] | 前区间: [{state['prev_zone_bottom']:.1f}, {state['prev_zone_top']:.1f}] ")
            else:
                it, ib = min(state["curr_zone_top"], state["pen_top"]), max(state["curr_zone_bottom"],
                                                                            state["pen_bottom"])
                if it > ib:
                    state["curr_zone_top"], state["curr_zone_bottom"] = it, ib
                    state["signal_ready"] = 0

                    print(f"[{last_dt}] | 区间监控： atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] | 前区间: [{state['prev_zone_bottom']:.1f}, {state['prev_zone_top']:.1f}] ")
                else:
                    if state["pen_type"] == "YIN" and state["pen_bottom"] > state["curr_zone_top"]:
                        state["signal_ready"] = 1
                    elif state["pen_type"] == "YANG" and state["pen_top"] < state["curr_zone_bottom"]:
                        state["signal_ready"] = -1

            # 【环节 A：结构破坏离场】
            if (state["signal_ready"] == 1 and is_yang) or (state["signal_ready"] == -1 and is_yin):
                state["prev_zone_top"], state["prev_zone_bottom"] = state["curr_zone_top"], state["curr_zone_bottom"]
                v1 = "空头离场" if is_yang else "多头离场"
                exit_reason = "区间结构切换"

                if is_yang:
                    state["count_up_zones"] += 1;
                    state["count_down_zones"] = 0
                else:
                    state["count_down_zones"] += 1;
                    state["count_up_zones"] = 0
                state["curr_zone_top"], state["curr_zone_bottom"] = state["pen_top"], state["pen_bottom"]
                state["signal_ready"], state["count_above"], state["count_below"] = 0, 0, 0

                print(f"[{last_dt}] | 区间切换： atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] | 前区间: [{state['prev_zone_bottom']:.1f}, {state['prev_zone_top']:.1f}] ")

        state["pen_top"], state["pen_bottom"], state["pen_type"] = h, l, ("YANG" if is_yang else "YIN")
        # print(f"[{last_dt}] | 笔转换： | pen_type: [{state['pen_type']}, pen_top ： {state['pen_top']:.1f}] | pen_bottom: [{state['pen_bottom']:.1f}] ")
    else:
        if state["pen_type"] == "YANG":
            state["pen_top"] = max(state["pen_top"], h)

            # print(f"[{last_dt}] | 笔延续： | pen_type: [{state['pen_type']}, pen_top ： {state['pen_top']:.1f}] | pen_bottom: [{state['pen_bottom']:.1f}] ")
        else:
            state["pen_bottom"] = min(state["pen_bottom"], l)
            # print(f"[{last_dt}] | 笔延续： | pen_type: [{state['pen_type']}, pen_top ： {state['pen_top']:.1f}] | pen_bottom: [{state['pen_bottom']:.1f}] ")

        # --- 新增 4. 失效检查：如果预备信号出现后，价格又回到了区间内，则取消信号 ---
        if state["signal_ready"] == 1 and l <= state["curr_zone_top"]:
            state["signal_ready"] = 0
        if state["signal_ready"] == -1 and h >= state["curr_zone_bottom"]:
            state["signal_ready"] = 0

    # --- 5. 入场过滤逻辑 ---
    # 【重要修改】：将 bridge 获取提前到逻辑判断之前
    if symbol not in _STATE_BRIDGE:
        _STATE_BRIDGE[symbol] = {"long_active": False, "short_active": False, "locked_sl_bottom": 0.0,
                                 "locked_sl_top": 0.0, "entry_price": 0.0, "current_atr": 0.0}
    bridge = _STATE_BRIDGE[symbol]

    is_long_pos = bridge.get("long_active", False)
    is_short_pos = bridge.get("short_active", False)

    # --- 5. 入场过滤逻辑 ---
    if state["zone_initialized"]:
        # 提取当前 K 线的波幅，以及当前“笔”的波幅
        pen_size = h - l
        pen_size_1 = state["pen_top"] - state["pen_bottom"]

        # --- 完全复刻 MT5 的逻辑 A、B、C ---
        if l > state["curr_zone_top"]:
            # 只要满足：当前K线是大阳线，或者当前处于多头笔且笔的长度达标
            if (pen_size > current_atr and is_yang) or (state["pen_type"] == "YANG" and pen_size_1 > current_atr * 2 * ATR_M):
                state["count_above"] += 1
                state["count_below"] = 0
        elif h < state["curr_zone_bottom"]:
            # 只要满足：当前K线是大阴线，或者当前处于空头笔且笔的长度达标
            if (pen_size > current_atr and is_yin) or (state["pen_type"] == "YIN" and pen_size_1 > current_atr * 2 * ATR_M):
                state["count_below"] += 1
                state["count_above"] = 0
        else:
            # 逻辑 C：价格回到了区间内，计数彻底清零
            state["count_above"] = 0
            state["count_below"] = 0

        # --- 后续入场判定 ---
        if state["count_above"] >= 1:
            dist = c0 - state["curr_zone_top"]
            c_rsi, c_ma = current_rsi > RSI_U, c0 > current_ma
            c_dist = (state["count_up_zones"] == 0 and dist < 2 * IN_PY) or (dist < IN_PY)

            if c_rsi and c_ma and is_expanding and c_dist and not bridge.get("long_active", False):
                v1 = "多头入场"
                state["count_above"] = 0
            else:
                v1 = "空头离场"
                exit_reason = "向上突破1k"

        elif state["count_below"] >= 1:
            dist = state["curr_zone_bottom"] - c0
            c_rsi, c_ma = current_rsi < RSI_L, c0 < current_ma
            c_dist = (state["count_down_zones"] == 0 and dist < 2 * IN_PY) or (dist < IN_PY)

            if c_rsi and c_ma and is_expanding and c_dist and not bridge.get("short_active", False):
                v1 = "空头入场"
                state["count_below"] = 0
            else:
                v1 = "多头离场"
                exit_reason = "向下突破1k"

    # --- 6. 持仓与桥梁状态更新 ---
    # 无持仓屏蔽离场信号
    if v1 == "多头离场" and not bridge["long_active"]:
        v1 = "其他"
    elif v1 == "空头离场" and not bridge["short_active"]:
        v1 = "其他"

    if v1 == "多头入场":
        bridge["long_active"], bridge["short_active"] = True, False
        bridge["locked_sl_bottom"], bridge["entry_price"] = state["curr_zone_top"], c0
    elif v1 == "空头入场":
        bridge["short_active"], bridge["long_active"] = True, False
        bridge["locked_sl_top"], bridge["entry_price"] = state["curr_zone_bottom"], c0
    elif v1 == "多头离场":
        bridge["long_active"] = False
    elif v1 == "空头离场":
        bridge["short_active"] = False

    bridge["current_atr"] = current_atr
    c.cache["pen_zone_state"] = state
    res = create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)
    c.cache["last_processed_dt"], c.cache["last_v1_signal"] = last_dt, res

    if v1 != "其他":
        e_price = bridge['entry_price']
        reason_str = f" | 原因: {exit_reason}" if exit_reason else ""
        price_log = f"信号触发价:{c0:.1f}" if "入场" in v1 else f"平仓价:{c0:.1f} (开仓价:{e_price:.1f}){reason_str}"
        print(
            f"[{last_dt}] | 信号: {v1} | {price_log} | atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] | 前区间: [{state['prev_zone_bottom']:.1f}, {state['prev_zone_top']:.1f}] "
            f"| atr: {current_atr} ")

    return res


# def pos_pen_zone_sl_be_V1(c: CZSC, **kwargs) -> OrderedDict:
#     """
#     【环节 C：3分钟实时触价逻辑】
#     """
#
#
#     symbol, freq = c.symbol, c.freq.value
#     pos_name = kwargs.get("pos_name", "PenZoneV1")
#     k1, k2, k3 = freq, pos_name, "SLBEV1"
#     bridge = _STATE_BRIDGE.get(symbol, {"long_active": False, "short_active": False})
#     cur_price, v1 = c.bars_raw[-1].close, "其他"
#
#     if bridge.get("long_active"):
#         sl_bottom = bridge.get("locked_sl_bottom", 0.0)
#         if sl_bottom > 0 and cur_price <= sl_bottom:
#             v1 = "多头止损触发"
#             _STATE_BRIDGE[symbol]["long_active"] = False
#     elif bridge.get("short_active"):
#         sl_top = bridge.get("locked_sl_top", 0.0)
#         if sl_top > 0 and cur_price >= sl_top:
#             v1 = "空头止损触发"
#             _STATE_BRIDGE[symbol]["short_active"] = False
#
#     if v1 != "其他":
#         print(f"[{c.bars_raw[-1].dt}] 🚨 [持仓触价：{v1}] 现价:{cur_price:.1f} | 开仓价:{bridge.get('entry_price', 0.0):.1f}"
#               f"| 触碰价:{bridge.get('locked_sl_bottom', 0.0):.1f} | 触碰价:{bridge.get('locked_sl_top', 0.0):.1f}")
#     return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

def pos_pen_zone_sl_be_V1(c: CZSC, **kwargs) -> OrderedDict:
    """
    【环节 C：1分钟实时触价逻辑】
    """

    symbol, freq = c.symbol, c.freq.value
    pos_name = kwargs.get("pos_name", "PenZoneV1")
    k1, k2, k3 = freq, pos_name, "SLBEV1"
    bridge = _STATE_BRIDGE.get(symbol, {"long_active": False, "short_active": False})
    cur_price, v1 = c.bars_raw[-1].close, "其他"

    if bridge.get("long_active"):
        sl_bottom = bridge.get("locked_sl_bottom", 0.0)
        entry_price = bridge.get("entry_price", 0.0)
        atr = bridge.get("current_atr", 0.0)

        # --- 核心新增：盈利 2 倍 ATR 触发多头保本 ---
        if atr > 0 and entry_price > 0 and cur_price >= entry_price + 2 * atr:
            if sl_bottom < entry_price:
                _STATE_BRIDGE[symbol]["locked_sl_bottom"] = entry_price
                sl_bottom = entry_price  # 局部变量同步，用于下方止损判断

        if sl_bottom > 0 and cur_price <= sl_bottom:
            v1 = "多头止损触发"
            _STATE_BRIDGE[symbol]["long_active"] = False

    elif bridge.get("short_active"):
        sl_top = bridge.get("locked_sl_top", 0.0)
        entry_price = bridge.get("entry_price", 0.0)
        atr = bridge.get("current_atr", 0.0)

        # --- 核心新增：盈利 2 倍 ATR 触发空头保本 ---
        if atr > 0 and entry_price > 0 and cur_price <= entry_price - 2 * atr:
            if sl_top > entry_price or sl_top == 0:
                _STATE_BRIDGE[symbol]["locked_sl_top"] = entry_price
                sl_top = entry_price  # 局部变量同步，用于下方止损判断

        if sl_top > 0 and cur_price >= sl_top:
            v1 = "空头止损触发"
            _STATE_BRIDGE[symbol]["short_active"] = False

    if v1 != "其他":
        print(f"[{c.bars_raw[-1].dt}] 🚨 [持仓触价：{v1}] 现价:{cur_price:.1f} | 开仓价:{bridge.get('entry_price', 0.0):.1f}"
              f"| 触碰价:{bridge.get('locked_sl_bottom', 0.0):.1f} | 触碰价:{bridge.get('locked_sl_top', 0.0):.1f}")

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)





def range_trade_signal_V1(c: CZSC, di: int = 1, **kwargs) -> OrderedDict:
    """
    【集成版】3分钟区间震荡开仓逻辑 (包含独立的笔与区间计算)
    """
    freq = c.freq.value
    symbol = c.symbol
    k1, k2, k3 = freq, f"D{di}", "RangeV1"

    if len(c.bars_raw) < 251:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    # --- 1. 获取当前 K 线数据 ---
    last_bar = c.bars_raw[-1]
    last_dt = last_bar.dt
    h, l, o, c0 = last_bar.high, last_bar.low, last_bar.open, last_bar.close
    is_yang, is_yin = c0 > o, c0 < o

    # --- 2. 参数指标计算 ---
    ATR_N = int(kwargs.get("atr_period", 14))
    ATR_M = float(kwargs.get("atr_multiplier", 1.0))
    MA_N = int(kwargs.get("ma_period", 250))
    RSI_U, RSI_L = float(kwargs.get("rsi_upper", 55)), float(kwargs.get("rsi_lower", 45))

    relevant_bars = c.bars_raw[-MA_N:]
    close_arr = np.array([b.close for b in relevant_bars])
    high_arr = np.array([b.high for b in relevant_bars])
    low_arr = np.array([b.low for b in relevant_bars])

    # ATR 计算
    tr = np.maximum(high_arr[1:] - low_arr[1:],
                    np.maximum(np.abs(high_arr[1:] - close_arr[:-1]), np.abs(low_arr[1:] - close_arr[:-1])))
    current_atr = np.mean(tr[-ATR_N:])

    # RSI 计算
    delta = np.diff(close_arr)
    gain, loss = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    avg_loss = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    current_rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))

    # MA & BOLL 计算
    current_ma = np.mean(close_arr[-MA_N:])
    bb_mid = pd.Series(close_arr).rolling(window=20).mean()
    bb_std = pd.Series(close_arr).rolling(window=20).std()
    is_expanding = (bb_mid + 2 * bb_std).diff().iloc[-1] > 0

    # --- 3. 笔与区间状态维护 (使用独立的 range 缓存键，避免与原策略冲突) ---
    state = c.cache.get("range_pen_zone_state", {
        "pen_type": "NONE", "pen_top": 0.0, "pen_bottom": 0.0,
        "zone_initialized": False, "curr_zone_top": 0.0, "curr_zone_bottom": 0.0,
        "signal_ready": 0
    })

    # --- 4. 核心：整合进来的“笔与区间”计算逻辑 ---
    if not (is_yang or is_yin):
        pass
    elif state["pen_type"] == "NONE":
        state["pen_type"], state["pen_top"], state["pen_bottom"] = ("YANG" if is_yang else "YIN"), h, l
    elif (state["pen_type"] == "YANG" and is_yin) or (state["pen_type"] == "YIN" and is_yang):
        pen_size = state["pen_top"] - state["pen_bottom"]

        # 笔幅达标，更新或建立区间
        if pen_size > current_atr * 2 * ATR_M:
            if not state["zone_initialized"]:
                state["curr_zone_top"], state["curr_zone_bottom"] = state["pen_top"], state["pen_bottom"]
                state["zone_initialized"] = True

                print(f"[{last_dt}] | 区间监控： atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] ")
            else:
                # 区间重叠取交集（收缩）
                it, ib = min(state["curr_zone_top"], state["pen_top"]), max(state["curr_zone_bottom"], state["pen_bottom"])
                if it > ib:
                    state["curr_zone_top"], state["curr_zone_bottom"] = it, ib
                    state["signal_ready"] = 0

                    print(f"[{last_dt}] | 区间监控： atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] ")
                else:
                    if state["pen_type"] == "YIN" and state["pen_bottom"] > state["curr_zone_top"]:
                        state["signal_ready"] = 1
                    elif state["pen_type"] == "YANG" and state["pen_top"] < state["curr_zone_bottom"]:
                        state["signal_ready"] = -1

        state["pen_top"], state["pen_bottom"], state["pen_type"] = h, l, ("YANG" if is_yang else "YIN")
        # print(f"[{last_dt}] | 笔转换： | pen_type: [{state['pen_type']}, pen_top ： {state['pen_top']:.1f}] | pen_bottom: [{state['pen_bottom']:.1f}] ")
    else:
        if state["pen_type"] == "YANG":
            state["pen_top"] = max(state["pen_top"], h)
            # print(f"[{last_dt}] | 笔延续： | pen_type: [{state['pen_type']}, pen_top ： {state['pen_top']:.1f}] | pen_bottom: [{state['pen_bottom']:.1f}] ")
        else:
            state["pen_bottom"] = min(state["pen_bottom"], l)
            # print(f"[{last_dt}] | 笔延续： | pen_type: [{state['pen_type']}, pen_top ： {state['pen_top']:.1f}] | pen_bottom: [{state['pen_bottom']:.1f}] ")

        # --- 新增 4. 失效检查：如果预备信号出现后，价格又回到了区间内，则取消信号 ---
        if state["signal_ready"] == 1 and l <= state["curr_zone_top"]:
            state["signal_ready"] = 0
        if state["signal_ready"] == -1 and h >= state["curr_zone_bottom"]:
            state["signal_ready"] = 0

    # 把算好的状态存回 c.cache
    c.cache["range_pen_zone_state"] = state

    # --- 5. 震荡入场过滤逻辑 ---
    if "range_bridge" not in _STATE_BRIDGE:
         _STATE_BRIDGE["range_bridge"] = {symbol: {"long_active": False, "short_active": False, "entry_price": 0.0}}
    if symbol not in _STATE_BRIDGE["range_bridge"]:
         _STATE_BRIDGE["range_bridge"][symbol] = {"long_active": False, "short_active": False, "entry_price": 0.0}

    r_bridge = _STATE_BRIDGE["range_bridge"][symbol]
    v1 = "其他"

    r_bridge["current_atr"] = current_atr

    # 只有区间初始化成功后，才开始捕捉震荡机会
    if state["zone_initialized"]:
        zone_top, zone_bottom = state["curr_zone_top"], state["curr_zone_bottom"]

        # 辅助指标过滤（BOLL, RSI, MA）
        c_boll_short = c0 > bb_mid
        c_rsi_short = current_rsi > RSI_L
        c_ma_short = c0 < current_ma

        c_boll_long = c0 < bb_mid
        c_rsi_long = current_rsi < RSI_U
        c_ma_long = c0 > current_ma

        # 判定：1. 区间上轨附近(1倍ATR缓冲)转阴开空
        if not r_bridge["short_active"] and (c0 > zone_top):
            if is_yin :
                    # and c_boll_short and c_rsi_short and c_ma_short:
                v1 = "震荡开空"
                r_bridge["short_active"], r_bridge["long_active"] = True, False
                r_bridge["entry_price"] = c0
                # 【锁定止盈位】：开空时，止盈固定为当时的区间下轨
                r_bridge["tp_price"] = zone_top

        # 判定：2. 区间下轨附近(1倍ATR缓冲)转阳开多
        elif not r_bridge["long_active"] and (c0 < zone_bottom):
            if is_yang :
                    # and c_boll_long and c_rsi_long and c_ma_long:
                v1 = "震荡开多"
                r_bridge["long_active"], r_bridge["short_active"] = True, False
                r_bridge["entry_price"] = c0
                # 【锁定止盈位】：开多时，止盈固定为当时的区间上轨
                r_bridge["tp_price"] = zone_bottom

    c.cache["last_processed_dt"] = last_dt

    if v1 != "其他":
        print(
            f"[{last_dt}] | 信号: {v1} | {r_bridge['entry_price']:.1f} | atr : {current_atr} | 区间: [{state['curr_zone_bottom']:.1f}, {state['curr_zone_top']:.1f}] ")

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


def pos_range_sl_tp_V1(c: CZSC, **kwargs) -> OrderedDict:
    """
    【修复版】触价止盈止损：移除无效的 positions 依赖，加入平仓冷却印记
    """
    symbol = c.symbol
    pos_name = kwargs.get("pos_name", "PenZoneV1")
    k1, k2, k3 = c.freq.value, pos_name, "RangeSLBEV1"

    if "range_bridge" not in _STATE_BRIDGE or symbol not in _STATE_BRIDGE["range_bridge"]:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    r_bridge = _STATE_BRIDGE["range_bridge"][symbol]
    atr = r_bridge.get("current_atr", 0.0)
    cur_price = c.bars_raw[-1].close
    v1 = "其他"

    # 直接从 bridge 获取开仓时锁定的价格
    entry_price = r_bridge.get("entry_price", 0.0)
    tp_price = r_bridge.get("tp_price", 0.0)  # 这里就是你开仓时刻的 zone_top/bottom


    is_long = r_bridge.get("long_active", False)
    is_short = r_bridge.get("short_active", False)

    if is_long:
        # 使用锁定的 tp_price 止盈
        if tp_price > 0 and cur_price >= tp_price:
            v1 = "震荡多头止盈"
            r_bridge["long_active"] = False
        elif cur_price <= entry_price - 2 * atr:
            v1 = "震荡多头止损"
            r_bridge["long_active"] = False

    elif is_short:
        # 使用锁定的 tp_price 止盈
        if tp_price > 0 and cur_price <= tp_price:
            v1 = "震荡空头止盈"
            r_bridge["short_active"] = False
        elif cur_price >= entry_price + 2 * atr:
            v1 = "震荡空头止损"
            r_bridge["short_active"] = False

    if v1 != "其他":
        # 【关键修复】：记录平仓时的 K线时间戳，防止开仓函数在同一根 K线上立刻反手
        r_bridge["last_exit_dt"] = c.bars_raw[-1].dt
        print(f"[{c.bars_raw[-1].dt}] 🎯 [发出平仓信号：{v1}] 现价:{cur_price:.1f} | 目标:{tp_price:.1f}")

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

def get_events_range_trade(trigger_freq="1分钟") -> List[Event]:
    """新增震荡系统的事件映射"""
    s_long_in = Signal("3分钟_D1_RangeV1_震荡开多_任意_任意_0")
    s_short_in = Signal("3分钟_D1_RangeV1_震荡开空_任意_任意_0")
    s_long_tp = Signal(f"{trigger_freq}_RangeTradeV1_RangeSLBEV1_震荡多头止盈_任意_任意_0")
    s_long_sl = Signal(f"{trigger_freq}_RangeTradeV1_RangeSLBEV1_震荡多头止损_任意_任意_0")
    s_short_tp = Signal(f"{trigger_freq}_RangeTradeV1_RangeSLBEV1_震荡空头止盈_任意_任意_0")
    s_short_sl = Signal(f"{trigger_freq}_RangeTradeV1_RangeSLBEV1_震荡空头止损_任意_任意_0")

    return [
        Event(name="震荡开多", operate=Operate.LO, factors=[Factor(name="入场", signals_all=[s_long_in])]),
        Event(name="震荡开空", operate=Operate.SO, factors=[Factor(name="入场", signals_all=[s_short_in])]),
        Event(name="震荡平多", operate=Operate.LE, factors=[
            Factor(name="止盈", signals_all=[s_long_tp]),
            Factor(name="止损", signals_all=[s_long_sl]),
        ]),
        Event(name="震荡平空", operate=Operate.SE, factors=[
            Factor(name="止盈", signals_all=[s_short_tp]),
            Factor(name="止损", signals_all=[s_short_sl]),
        ])
    ]






def get_events_pen_zone(trigger_freq="1分钟") -> List[Event]:
    s_long_in = Signal("3分钟_D1_PenZoneV1_多头入场_任意_任意_0")
    s_short_in = Signal("3分钟_D1_PenZoneV1_空头入场_任意_任意_0")
    s_long_exit = Signal("3分钟_D1_PenZoneV1_多头离场_任意_任意_0")
    s_short_exit = Signal("3分钟_D1_PenZoneV1_空头离场_任意_任意_0")
    s_long_sl = Signal(f"{trigger_freq}_PenZoneV1_SLBEV1_多头止损触发_任意_任意_0")
    s_short_sl = Signal(f"{trigger_freq}_PenZoneV1_SLBEV1_空头止损触发_任意_任意_0")

    return [
        Event(name="开多", operate=Operate.LO, factors=[Factor(name="入场", signals_all=[s_long_in])]),
        Event(name="开空", operate=Operate.SO, factors=[Factor(name="入场", signals_all=[s_short_in])]),
        Event(name="平多", operate=Operate.LE, factors=[
            Factor(name="正常离场", signals_all=[s_long_exit]),
            Factor(name="实时止损", signals_all=[s_long_sl]),
            Factor(name="反手", signals_all=[s_short_in]),
        ]),
        Event(name="平空", operate=Operate.SE, factors=[
            Factor(name="正常离场", signals_all=[s_short_exit]),
            Factor(name="实时止损", signals_all=[s_short_sl]),
            Factor(name="反手", signals_all=[s_long_in]),
        ])
    ]


class PenZoneStrategy(CzscStrategyBase):
    # trigger_freq = "3分钟"
    #
    # @property
    # def positions(self) -> List[Position]:
    #     e = get_events_pen_zone(self.trigger_freq)
    #     return [Position(symbol=self.symbol, name="PenZoneV1", opens=[e[0], e[1]], exits=[e[2], e[3]], T0=True)]
    #
    # @property
    # def signals_config(self):
    #     return [
    #         {"name": pen_zone_signal_V1, "freq": "3分钟", "di": 1},
    #         {"name": pos_pen_zone_sl_be_V1, "freq": self.trigger_freq, "pos_name": "PenZoneV1"}
    #     ]

    trigger_freq = "1分钟"

    @property
    def positions(self) -> List[Position]:
        # e1 = get_events_pen_zone(self.trigger_freq)  # 原突破系统
        e2 = get_events_range_trade(self.trigger_freq)  # 新震荡系统

        # 将两套系统的仓位逻辑并列挂载，互不干扰
        return [
             # Position(symbol=self.symbol, name="PenZoneV1", opens=[e1[0], e1[1]], exits=[e1[2], e1[3]], T0=True),
            Position(symbol=self.symbol, name="RangeTradeV1", opens=[e2[0], e2[1]], exits=[e2[2], e2[3]], T0=True)
        ]

    @property
    def signals_config(self):
        return [
            # {"name": pen_zone_signal_V1, "freq": "3分钟", "di": 1},
            # {"name": pos_pen_zone_sl_be_V1, "freq": self.trigger_freq, "pos_name": "PenZoneV1"},

            # 挂载新震荡策略的信号与触价
            {"name": range_trade_signal_V1, "freq": "3分钟", "di": 1},
            {"name": pos_range_sl_tp_V1, "freq": self.trigger_freq, "pos_name": "RangeTradeV1"}
        ]

