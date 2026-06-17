# -*- coding: utf-8 -*-
"""
author: 编码助手
describe: 笔压缩区间 + 连续脱离 的 Pen_Zone 单品种策略骨架 (严格对齐 MT5 逻辑版)
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, OrderedDict
import numpy as np
import pandas as pd
from czsc import CZSC, Direction, CzscStrategyBase, CzscTrader, Freq, Operate, Signal, Factor, Event, Position
from czsc.utils import create_single_signal
from czsc.traders.base import CzscSignals

import logging

logger = logging.getLogger(__name__)

def _log_mt5(c: CZSC, tag: str, msg: str, **kv):
    """记录可回放的执行日志（用于逐条对照 MT5 Print 输出）

    - 写入 c.cache['pen_zone_mt5_logs']（环形列表，默认最多 5000 条）
    - 同时用 logger.info 输出（便于直接看控制台/文件日志）
    """
    logs = c.cache.get("pen_zone_mt5_logs", [])
    max_len = int(c.cache.get("pen_zone_mt5_logs_max_len", 5000) or 5000)

    dt = None
    try:
        if c.bars_raw:
            dt = getattr(c.bars_raw[-1], "dt", None)
    except Exception:
        dt = None

    item = {"dt": dt, "tag": tag, "msg": msg}
    if kv:
        item.update(kv)
    logs.append(item)
    if len(logs) > max_len:
        logs = logs[-max_len:]
    c.cache["pen_zone_mt5_logs"] = logs

    text = f"[PenZoneMt5] {tag} {msg}" + (f" | {kv}" if kv else "")
    try:
        # INFO 级别更规范，但很多用户环境默认 WARNING，导致看不到输出
        logger.info(text)
    except Exception:
        pass

    # 兜底：如果 logging 没配置 handler / 或被级别过滤，则直接打印到控制台，方便逐条对照 MT5
    try:
        root = logging.getLogger()
        if (not logger.hasHandlers() and not root.hasHandlers()) or (root.level > logging.INFO and logger.level in [0, logging.NOTSET]):
            print(text)
    except Exception:
        pass

def _is_bollinger_expanding(close_s: pd.Series, bb_period: int, bb_dev: float, period: int = 5) -> bool:
    """对齐 MT5: IsBollingerExpanding() -> 返回 is_growing (带宽是否增长)"""
    if len(close_s) < max(bb_period + 2, period + 2):
        return False

    bb_mid = close_s.rolling(window=bb_period, min_periods=1).mean()
    bb_std = close_s.rolling(window=bb_period, min_periods=1).std()
    bb_upper = bb_mid + bb_dev * bb_std
    bb_lower = bb_mid - bb_dev * bb_std

    # MT5 版本最后 return (is_growing)
    cur_w = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    pre_w = float(bb_upper.iloc[-2] - bb_lower.iloc[-2])
    return cur_w > pre_w


def _get_dynamic_atr_multiplier(close_s: pd.Series,
                                bb_period: int,
                                bb_dev: float,
                                period: int = 5,
                                base_multiplier: float = 1.0,
                                max_multiplier: float = 4.0) -> float:
    """对齐 MT5: GetDynamicATRMultiplier()

    基于布林带带宽的“缩口程度”调整 ATR 过滤倍数：
    width_ratio = average_width / current_width
    dynamic = base * width_ratio, 并限制到 [base, max]
    """
    if len(close_s) < max(bb_period + 1, period + 1):
        return float(base_multiplier)

    bb_mid = close_s.rolling(window=bb_period, min_periods=1).mean()
    bb_std = close_s.rolling(window=bb_period, min_periods=1).std()
    bb_upper = bb_mid + bb_dev * bb_std
    bb_lower = bb_mid - bb_dev * bb_std

    widths = (bb_upper - bb_lower).iloc[-period:]
    current_width = float(widths.iloc[-1])
    if current_width <= 0:
        return float(max_multiplier)

    average_width = float(widths.mean())
    width_ratio = average_width / current_width
    dynamic = float(base_multiplier) * float(width_ratio)
    dynamic = max(float(base_multiplier), min(dynamic, float(max_multiplier)))
    return float(np.round(dynamic, 2))


def pen_zone_signal_mt5_v1(c: CZSC, di: int = 1, **kwargs) -> OrderedDict:
    """Pen_Zone 笔压缩区间突破信号 (严格对齐 MQL5 版本)

    注意：在 `signals_config` 中指定了 `freq` 时，该函数会收到对应周期的 CZSC 对象作为第一个参数。
    """
    if di != 1:
        raise ValueError("This signal logic must run on the last closed bar (di=1)")

    # --- 信号基础定义 ---
    k1 = str(getattr(c, "freq", kwargs.get("k1", "15分钟")))
    k2 = f"D{di}"
    k3 = "PenZoneMt5V1"

    if len(c.bars_raw) < 251:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="数据不足")

    last_bar = c.bars_raw[-1]
    h, l, o, c0 = last_bar.high, last_bar.low, last_bar.open, last_bar.close
    is_yang = c0 > o
    is_yin = c0 < o
    log_on = bool(kwargs.get("log", True))

    # --- 参数获取 (对齐 MT5) ---
    ATR_N = int(kwargs.get("atr_period", 14))
    MA_N = int(kwargs.get("ma_period", 360))
    RSI_N = int(kwargs.get("rsi_period", 14))
    RSI_U = float(kwargs.get("rsi_upper", 45))
    RSI_L = float(kwargs.get("rsi_lower", 55))
    BB_N = int(kwargs.get("bb_period", 20))
    BB_K = float(kwargs.get("bb_dev", 2.0))
    MA_FILTER = bool(kwargs.get("use_ma_filter", True))
    IN_PY = float(kwargs.get("in_py", 100))
    INOUT_COUNT = int(kwargs.get("inout_count", 1))
    BASE_ATR_M = float(kwargs.get("base_atr_multiplier", 1.0))
    MAX_ATR_M = float(kwargs.get("max_atr_multiplier", 4.0))
    IN_PERIOD = int(kwargs.get("in_period", 170))

    # --- 技术指标计算 ---
    df = pd.DataFrame([{"high": b.high, "low": b.low, "close": b.close} for b in c.bars_raw])
    high_s, low_s, close_s = df["high"], df["low"], df["close"]

    tr = pd.concat([high_s - low_s, (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=ATR_N, min_periods=1).mean()
    current_atr = atr.iloc[-1] if not atr.empty else 0.0

    delta = close_s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/RSI_N, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/RSI_N, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0

    ma = close_s.rolling(window=MA_N, min_periods=1).mean()
    current_ma = ma.iloc[-1] if not ma.empty else 0.0

    # --- 对齐 MT5: 布林带扩张仅用“带宽增长”(is_growing) ---
    bb_ok = _is_bollinger_expanding(close_s, bb_period=BB_N, bb_dev=BB_K, period=5)

    # --- 对齐 MT5: 动态 ATR 过滤倍数 (基于布林带带宽缩口) ---
    dynamic_atr_multi = _get_dynamic_atr_multiplier(
        close_s,
        bb_period=BB_N,
        bb_dev=BB_K,
        period=IN_PERIOD,
        base_multiplier=BASE_ATR_M,
        max_multiplier=MAX_ATR_M,
    )

    # --- 状态缓存 (对齐 MT5) ---
    state = c.cache.get("pen_zone_mt5_state", {})
    if not state:
        state = {
            "pen_type": "NONE", "pen_top": 0.0, "pen_bottom": 0.0,
            "zone_initialized": False, "curr_zone_top": 0.0, "curr_zone_bottom": 0.0,
            "signal_ready": 0, "count_above": 0, "count_below": 0,
            "count_up_zones": 0, "count_down_zones": 0, # 新增：区间变换计数器
            # MT5: 焊死防线（对齐 locked edges）
            "locked_buy_edge": 0.0, "locked_sell_edge": 0.0,
            # MT5: 主力离场后的对冲保护边界（对齐“推止损到区间边缘”的效果，用事件近似）
            # dir: "BUY" 对冲多，"SELL" 对冲空，"NONE" 无
            "hedge_protect_dir": "NONE", "hedge_protect_edge": 0.0,
        }

    pen_type = state["pen_type"]
    pen_top, pen_bottom = state["pen_top"], state["pen_bottom"]
    zone_initialized = state["zone_initialized"]
    curr_top, curr_bottom = state["curr_zone_top"], state["curr_zone_bottom"]
    signal_ready = state["signal_ready"]
    count_above, count_below = state["count_above"], state["count_below"]
    count_up_zones, count_down_zones = state["count_up_zones"], state["count_down_zones"]
    locked_buy_edge = state.get("locked_buy_edge", 0.0)
    locked_sell_edge = state.get("locked_sell_edge", 0.0)
    hedge_protect_dir = state.get("hedge_protect_dir", "NONE")
    hedge_protect_edge = state.get("hedge_protect_edge", 0.0)
    
    open_long_signal, open_short_signal = False, False
    exit_long_signal, exit_short_signal = False, False

    if not (is_yang or is_yin):
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    if log_on:
        _log_mt5(
            c, "BAR",
            "bar inputs",
            h=float(h), l=float(l), o=float(o), close=float(c0),
            is_yang=bool(is_yang), is_yin=bool(is_yin),
            atr=float(current_atr), rsi=float(current_rsi), ma=float(current_ma),
            bb_ok=bool(bb_ok), dynamic_atr_multi=float(dynamic_atr_multi),
        )

    # --- 核心逻辑 1: 笔状态与区间变换 (UpdatePenState) ---
    if pen_type == "NONE":
        pen_type = "YANG" if is_yang else "YIN"
        pen_top, pen_bottom = h, l
        if log_on:
            _log_mt5(c, "PEN_INIT", "init first pen", pen_type=pen_type, pen_top=float(pen_top), pen_bottom=float(pen_bottom))
    elif (pen_type == "YANG" and is_yin) or (pen_type == "YIN" and is_yang):
        pen_size_1 = pen_top - pen_bottom

        # 对齐 MT5: 过滤：笔长度 < ATR * 动态倍数 -> 忽略此笔
        if not (current_atr > 0 and pen_size_1 < current_atr * dynamic_atr_multi):
            if not zone_initialized:
                curr_top, curr_bottom = pen_top, pen_bottom
                zone_initialized = True
                if log_on:
                    _log_mt5(c, "ZONE_INIT", "init zone by first pen", curr_top=float(curr_top), curr_bottom=float(curr_bottom))
            else:
                # 压缩逻辑 (OnPenCompleted)
                intersect_top = min(curr_top, pen_top)
                intersect_bottom = max(curr_bottom, pen_bottom)
                if intersect_top > intersect_bottom:
                    curr_top, curr_bottom = intersect_top, intersect_bottom
                    signal_ready = 0
                    if log_on:
                        _log_mt5(c, "ZONE_COMPRESS", "zone intersection exists -> compress", curr_top=float(curr_top), curr_bottom=float(curr_bottom))
                else:
                    if pen_type == "YIN" and pen_bottom > curr_top:
                        signal_ready = 1
                        if log_on:
                            _log_mt5(c, "PREP_LONG", "yin pen above zone -> wait yang", pen_bottom=float(pen_bottom), curr_top=float(curr_top))
                    elif pen_type == "YANG" and pen_top < curr_bottom:
                        signal_ready = -1
                        if log_on:
                            _log_mt5(c, "PREP_SHORT", "yang pen below zone -> wait yin", pen_top=float(pen_top), curr_bottom=float(curr_bottom))
            
            # 转色入场确立
            if signal_ready == 1 and is_yang:
                curr_top, curr_bottom = pen_top, pen_bottom
                signal_ready = 0
                count_above, count_below = 0, 0
                count_up_zones += 1
                count_down_zones = 0
                exit_short_signal = True
                if log_on:
                    _log_mt5(c, "ZONE_SHIFT_UP", "prep long confirmed -> shift zone", curr_top=float(curr_top), curr_bottom=float(curr_bottom),
                             count_up_zones=int(count_up_zones))
            elif signal_ready == -1 and is_yin:
                curr_top, curr_bottom = pen_top, pen_bottom
                signal_ready = 0
                count_above, count_below = 0, 0
                count_down_zones += 1
                count_up_zones = 0
                exit_long_signal = True
                if log_on:
                    _log_mt5(c, "ZONE_SHIFT_DOWN", "prep short confirmed -> shift zone", curr_top=float(curr_top), curr_bottom=float(curr_bottom),
                             count_down_zones=int(count_down_zones))

        else:
            if log_on:
                _log_mt5(c, "PEN_FILTER", "pen too small vs ATR threshold -> ignore", pen_size=float(pen_size_1),
                         atr=float(current_atr), dynamic_atr_multi=float(dynamic_atr_multi))

        pen_top, pen_bottom = h, l
        pen_type = "YANG" if is_yang else "YIN"
        if log_on:
            _log_mt5(c, "PEN_SWITCH", "pen color reversed -> start new pen", pen_type=pen_type, pen_top=float(pen_top), pen_bottom=float(pen_bottom))
    else:
        # 笔延续
        if pen_type == "YANG":
            pen_top = max(pen_top, h)
        else:
            pen_bottom = min(pen_bottom, l)
        if log_on:
            _log_mt5(c, "PEN_EXTEND", "pen continues", pen_type=pen_type, pen_top=float(pen_top), pen_bottom=float(pen_bottom))
        
        # 信号失效检查
        if signal_ready == 1 and l <= curr_top:
            signal_ready = 0
            if log_on:
                _log_mt5(c, "PREP_CANCEL", "prep long cancelled by re-enter zone", l=float(l), curr_top=float(curr_top))
        if signal_ready == -1 and h >= curr_bottom:
            signal_ready = 0
            if log_on:
                _log_mt5(c, "PREP_CANCEL", "prep short cancelled by re-enter zone", h=float(h), curr_bottom=float(curr_bottom))

    # --- 核心逻辑 2: 连续脱离区间检查 (CheckContinuousBreakout) ---
    if zone_initialized:
        pen_size = h - l
        pen_size_1 = pen_top - pen_bottom
        
        if l > curr_top:
            if (pen_size > current_atr and is_yang) or (pen_type == "YANG" and pen_size_1 > current_atr * dynamic_atr_multi):
                count_above += 1
                count_below = 0
                if log_on:
                    _log_mt5(c, "COUNT_ABOVE", "bar above zone counted", count_above=int(count_above), curr_top=float(curr_top))
        elif h < curr_bottom:
            if (pen_size > current_atr and is_yin) or (pen_type == "YIN" and pen_size_1 > current_atr * dynamic_atr_multi):
                count_below += 1
                count_above = 0
                if log_on:
                    _log_mt5(c, "COUNT_BELOW", "bar below zone counted", count_below=int(count_below), curr_bottom=float(curr_bottom))
        else:
            count_above, count_below = 0, 0
            if log_on:
                _log_mt5(c, "COUNT_RESET", "price returned into zone -> reset counts", curr_top=float(curr_top), curr_bottom=float(curr_bottom))

        # 执行交易决策 (严格对齐 MT5)
        if count_above >= INOUT_COUNT:
            exit_short_signal = True
            if current_rsi > RSI_U and (not MA_FILTER or c0 > current_ma) and bb_ok:
                aa = c0 - curr_top
                if (count_up_zones == 0 and aa < 2 * IN_PY) or (aa < IN_PY and count_up_zones < 4):
                    open_long_signal = True
                    count_above = 0 
                    # MT5: 开多单时锁定防线 = 区间底
                    locked_buy_edge = float(curr_bottom)
                    locked_sell_edge = 0.0
                    if log_on:
                        _log_mt5(c, "OPEN_LONG", "breakout long confirmed", aa=float(aa), locked_buy_edge=float(locked_buy_edge),
                                 count_up_zones=int(count_up_zones))
            elif log_on:
                _log_mt5(c, "OPEN_LONG_BLOCK", "breakout long blocked", rsi=float(current_rsi), rsi_u=float(RSI_U),
                         ma_filter=bool(MA_FILTER), bb_ok=bool(bb_ok))

        if count_below >= INOUT_COUNT:
            exit_long_signal = True
            if current_rsi < RSI_L and (not MA_FILTER or c0 < current_ma) and bb_ok:
                aa = curr_bottom - c0
                if (count_down_zones == 0 and aa < 2 * IN_PY) or (aa < IN_PY and count_down_zones < 4):
                    open_short_signal = True
                    count_below = 0
                    # MT5: 开空单时锁定防线 = 区间顶
                    locked_sell_edge = float(curr_top)
                    locked_buy_edge = 0.0
                    if log_on:
                        _log_mt5(c, "OPEN_SHORT", "breakout short confirmed", aa=float(aa), locked_sell_edge=float(locked_sell_edge),
                                 count_down_zones=int(count_down_zones))
            elif log_on:
                _log_mt5(c, "OPEN_SHORT_BLOCK", "breakout short blocked", rsi=float(current_rsi), rsi_l=float(RSI_L),
                         ma_filter=bool(MA_FILTER), bb_ok=bool(bb_ok))

    # --- 更新缓存 ---
    state.update({
        "pen_type": pen_type, "pen_top": float(pen_top), "pen_bottom": float(pen_bottom),
        "zone_initialized": zone_initialized, "curr_zone_top": float(curr_top), 
        "curr_zone_bottom": float(curr_bottom), "signal_ready": int(signal_ready),
        "count_above": int(count_above), "count_below": int(count_below),
        "count_up_zones": int(count_up_zones), "count_down_zones": int(count_down_zones),
        "locked_buy_edge": float(locked_buy_edge),
        "locked_sell_edge": float(locked_sell_edge),
        "hedge_protect_dir": str(hedge_protect_dir),
        "hedge_protect_edge": float(hedge_protect_edge),
    })
    c.cache["pen_zone_mt5_state"] = state

    # --- 信号解析 ---
    if open_long_signal and exit_short_signal: v1 = "平空开多"
    elif open_short_signal and exit_long_signal: v1 = "平多开空"
    elif open_long_signal: v1 = "多信号"
    elif open_short_signal: v1 = "空信号"
    elif exit_long_signal: v1 = "多头离场"
    elif exit_short_signal: v1 = "空头离场"
    else: v1 = "其他"
    
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


def pen_zone_hedge_signal_mt5_v1(cat: CzscSignals, di: int = 1, **kwargs) -> OrderedDict:
    """MT5 对齐：焊死防线 + 回撤触发对冲 + 2ATR 保本后平对冲并解除防线

    说明：
    - 主力仓位：pos_name（默认 PenZoneMt5V1）
    - 对冲仓位：hedge_pos_name（默认 PenZoneMt5HedgeV1）
    - 防线来自 `pen_zone_mt5_state` 的 locked_buy_edge / locked_sell_edge
    """
    if di != 1:
        raise ValueError("This signal logic must run on the last closed bar (di=1)")

    freq = kwargs.get("base_freq", "15分钟")
    c: CZSC = cat.kas[freq]
    pos_name = kwargs.get("pos_name", "PenZoneMt5V1")
    hedge_pos_name = kwargs.get("hedge_pos_name", "PenZoneMt5HedgeV1")

    k1 = freq
    k2 = f"D{di}"
    k3 = "PenZoneMt5HedgeV1"

    if len(c.bars_raw) < 50:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    last_bar = c.bars_raw[-1]
    last_dt = getattr(last_bar, "dt", None)
    close = float(last_bar.close)
    log_on = bool(kwargs.get("log", False))

    # --- ATR (对齐 MT5: 用于 2ATR 触发一阶保本) ---
    ATR_N = int(kwargs.get("atr_period", 14))
    atr_be_multiplier = float(kwargs.get("atr_be_multiplier", 2.0))
    df = pd.DataFrame([{"high": b.high, "low": b.low, "close": b.close} for b in c.bars_raw])
    high_s, low_s, close_s = df["high"], df["low"], df["close"]
    tr = pd.concat([high_s - low_s, (high_s - close_s.shift(1)).abs(), (low_s - close_s.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=ATR_N, min_periods=1).mean()
    current_atr = float(atr.iloc[-1]) if len(atr) else 0.0

    # --- 读取状态缓存（焊死防线）---
    state = c.cache.get("pen_zone_mt5_state", {})
    locked_buy_edge = float(state.get("locked_buy_edge", 0.0) or 0.0)
    locked_sell_edge = float(state.get("locked_sell_edge", 0.0) or 0.0)
    hedge_protect_dir = str(state.get("hedge_protect_dir", "NONE") or "NONE")
    hedge_protect_edge = float(state.get("hedge_protect_edge", 0.0) or 0.0)
    curr_zone_top = float(state.get("curr_zone_top", 0.0) or 0.0)
    curr_zone_bottom = float(state.get("curr_zone_bottom", 0.0) or 0.0)

    # --- 读取主力仓位状态 ---
    main_pos = next((p for p in cat.positions if p.name == pos_name), None)
    main_dir = 0  # 1 多，-1 空，0 无
    main_entry = None
    if main_pos and main_pos.operates:
        last_op = main_pos.operates[-1]["op"]
        if last_op == Operate.LO:
            main_dir = 1
            main_entry = float(main_pos.operates[-1]["price"])
        elif last_op == Operate.SO:
            main_dir = -1
            main_entry = float(main_pos.operates[-1]["price"])

    # --- 读取对冲仓位状态 ---
    hedge_pos = next((p for p in cat.positions if p.name == hedge_pos_name), None)
    has_hedge_long = False
    has_hedge_short = False
    if hedge_pos and hedge_pos.operates:
        hop = hedge_pos.operates[-1]["op"]
        if hop == Operate.LO:
            has_hedge_long = True
        elif hop == Operate.SO:
            has_hedge_short = True

    v1 = "其他"

    # 场景 B（对齐 MT5 的可执行近似）：主力平仓成功 -> 记录对冲保护边界（顶/底）
    # 说明：MT5 会将遗留对冲单止损推到区间边缘；CZSC 无法移动止损价格，
    # 这里用“记录边界 + 触达边界时平对冲”来近似该行为。
    if main_pos and main_pos.operates and last_dt is not None:
        last_main_operate = main_pos.operates[-1]
        if last_main_operate.get("dt") == last_dt and last_main_operate.get("op") in [Operate.LE, Operate.SE]:
            # 主力平仓后，焊死防线失效
            state["locked_buy_edge"] = 0.0
            state["locked_sell_edge"] = 0.0

            # 若存在遗留对冲单，则设置保护边界：对冲空用区间底，对冲多用区间顶
            if has_hedge_short and curr_zone_bottom > 0:
                state["hedge_protect_dir"] = "SELL"
                state["hedge_protect_edge"] = float(curr_zone_bottom)
            elif has_hedge_long and curr_zone_top > 0:
                state["hedge_protect_dir"] = "BUY"
                state["hedge_protect_edge"] = float(curr_zone_top)
            else:
                state["hedge_protect_dir"] = "NONE"
                state["hedge_protect_edge"] = 0.0

            hedge_protect_dir = str(state["hedge_protect_dir"])
            hedge_protect_edge = float(state["hedge_protect_edge"])
            if log_on:
                _log_mt5(c, "HEDGE_PROTECT_SET", "main exit -> set hedge protect edge",
                         hedge_protect_dir=hedge_protect_dir, hedge_protect_edge=float(hedge_protect_edge),
                         curr_zone_top=float(curr_zone_top), curr_zone_bottom=float(curr_zone_bottom),
                         has_hedge_long=bool(has_hedge_long), has_hedge_short=bool(has_hedge_short))

    # 场景 A：主力多单回撤到防线 -> 开对冲空
    if main_dir == 1 and locked_buy_edge > 0 and close <= locked_buy_edge:
        if not has_hedge_short:
            v1 = "开对冲空"
            if log_on:
                _log_mt5(c, "HEDGE_OPEN_SELL", "price back to locked buy edge -> hedge sell", close=float(close), locked_buy_edge=float(locked_buy_edge))

    # 场景 A：主力空单回撤到防线 -> 开对冲多
    if main_dir == -1 and locked_sell_edge > 0 and close >= locked_sell_edge:
        if not has_hedge_long:
            v1 = "开对冲多"
            if log_on:
                _log_mt5(c, "HEDGE_OPEN_BUY", "price back to locked sell edge -> hedge buy", close=float(close), locked_sell_edge=float(locked_sell_edge))

    # 场景 C：一阶保本（主力浮盈达 2ATR）-> 平对冲 & 解除防线
    if main_dir == 1 and main_entry is not None and current_atr > 0:
        if close >= main_entry + atr_be_multiplier * current_atr:
            v1 = "平对冲"
            state["locked_buy_edge"] = 0.0
            if log_on:
                _log_mt5(c, "HEDGE_CLOSE", "2ATR reached -> close hedge", main_dir=1, entry=float(main_entry), close=float(close), atr=float(current_atr))
    if main_dir == -1 and main_entry is not None and current_atr > 0:
        if close <= main_entry - atr_be_multiplier * current_atr:
            v1 = "平对冲"
            state["locked_sell_edge"] = 0.0
            if log_on:
                _log_mt5(c, "HEDGE_CLOSE", "2ATR reached -> close hedge", main_dir=-1, entry=float(main_entry), close=float(close), atr=float(current_atr))

    # 场景 D：对冲保护边界触发 -> 平对冲
    # 对冲空：价格 <= 保护底边 -> 平对冲
    # 对冲多：价格 >= 保护顶边 -> 平对冲
    if hedge_protect_edge > 0:
        if hedge_protect_dir == "SELL" and has_hedge_short and close <= hedge_protect_edge:
            v1 = "平对冲"
            if log_on:
                _log_mt5(c, "HEDGE_CLOSE", "hit protect bottom -> close hedge", hedge_protect_edge=float(hedge_protect_edge), close=float(close))
        if hedge_protect_dir == "BUY" and has_hedge_long and close >= hedge_protect_edge:
            v1 = "平对冲"
            if log_on:
                _log_mt5(c, "HEDGE_CLOSE", "hit protect top -> close hedge", hedge_protect_edge=float(hedge_protect_edge), close=float(close))

    # 若触发平对冲，则清除保护边界（避免重复）
    if v1 == "平对冲":
        state["hedge_protect_dir"] = "NONE"
        state["hedge_protect_edge"] = 0.0
        if log_on:
            _log_mt5(c, "HEDGE_PROTECT_CLEAR", "clear hedge protect edge after close")

    c.cache["pen_zone_mt5_state"] = state
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


def pos_pen_zone_atr_be_mt5_v1(cat: CzscSignals, **kwargs) -> OrderedDict:
    """ATR 浮盈止损信号 (对齐 MT5: 盈利达 2倍 ATR 触发平仓)"""
    pos_name = kwargs.get("pos_name")
    f = kwargs.get("f", "15分钟") 
    atr_period = kwargs.get("atr_period", 14)
    atr_be_multiplier = kwargs.get("atr_be_multiplier", 2.0) 

    k1, k2, k3 = f, pos_name, "ATRBEV1"
    pos = next((p for p in cat.positions if p.name == pos_name), None)
    if not pos or not pos.operates or pos.operates[-1]['op'] not in [Operate.LO, Operate.SO]:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    c: CZSC = cat.kas[f]
    df = pd.DataFrame([{'high': b.high, 'low': b.low, 'close': b.close} for b in c.bars_raw])
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    current_atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    entry_price, current_price = pos.operates[-1]['price'], c.bars_raw[-1].close
    if pos.operates[-1]['op'] == Operate.LO and (current_price - entry_price >= current_atr * atr_be_multiplier):
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="多头止损")
    if pos.operates[-1]['op'] == Operate.SO and (entry_price - current_price >= current_atr * atr_be_multiplier):
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="空头止损")
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")


def get_events_pen_zone_mt5() -> List[Event]:
    """定义开平仓事件 (严格对齐 MT5 版本)"""
    s_long = Signal("15分钟_D1_PenZoneMt5V1_多信号_任意_任意_0")
    s_short = Signal("15分钟_D1_PenZoneMt5V1_空信号_任意_任意_0")
    s_reverse_to_long = Signal("15分钟_D1_PenZoneMt5V1_平空开多_任意_任意_0")
    s_reverse_to_short = Signal("15分钟_D1_PenZoneMt5V1_平多开空_任意_任意_0")
    s_exit_long = Signal("15分钟_D1_PenZoneMt5V1_多头离场_任意_任意_0")
    s_exit_short = Signal("15分钟_D1_PenZoneMt5V1_空头离场_任意_任意_0")

    return [
        Event(name="开多", operate=Operate.LO, factors=[
            Factor(name="F1", signals_all=[s_long]), 
           Factor(name="F1_R", signals_all=[s_reverse_to_long])
        ]),
        
        Event(name="开空", operate=Operate.SO, factors=[
            Factor(name="F2", signals_all=[s_short]), 
            Factor(name="F2_R", signals_all=[s_reverse_to_short])
        ]),
        
        Event(name="平多", operate=Operate.LE, factors=[
            Factor(name="F3_1", signals_all=[s_short]),
            Factor(name="F3_2", signals_all=[s_reverse_to_short]),
            Factor(name="F3_3", signals_all=[s_exit_long]),
        ]),
        
        Event(name="平空", operate=Operate.SE, factors=[
            Factor(name="F4_1", signals_all=[s_long]),
            Factor(name="F4_2", signals_all=[s_reverse_to_long]),
            Factor(name="F4_3", signals_all=[s_exit_short]),
        ])
    ]


def get_events_pen_zone_mt5_hedge() -> List[Event]:
    """对冲仓位事件：开对冲、平对冲（并被主力开仓强制平掉）"""
    s_hedge_open_short = Signal("15分钟_D1_PenZoneMt5HedgeV1_开对冲空_任意_任意_0")
    s_hedge_open_long = Signal("15分钟_D1_PenZoneMt5HedgeV1_开对冲多_任意_任意_0")
    s_hedge_close = Signal("15分钟_D1_PenZoneMt5HedgeV1_平对冲_任意_任意_0")

    # 主力一旦开仓/反手/离场，都强制把对冲平掉（对齐 MT5: 下单成功时 CloseAllPositionsByComment("Hedge")）
    s_main_long = Signal("15分钟_D1_PenZoneMt5V1_多信号_任意_任意_0")
    s_main_short = Signal("15分钟_D1_PenZoneMt5V1_空信号_任意_任意_0")
    s_main_reverse_to_long = Signal("15分钟_D1_PenZoneMt5V1_平空开多_任意_任意_0")
    s_main_reverse_to_short = Signal("15分钟_D1_PenZoneMt5V1_平多开空_任意_任意_0")
    s_main_exit_long = Signal("15分钟_D1_PenZoneMt5V1_多头离场_任意_任意_0")
    s_main_exit_short = Signal("15分钟_D1_PenZoneMt5V1_空头离场_任意_任意_0")

    return [
        Event(name="开对冲多", operate=Operate.LO, factors=[Factor(name="H1", signals_all=[s_hedge_open_long])]),
        Event(name="开对冲空", operate=Operate.SO, factors=[Factor(name="H2", signals_all=[s_hedge_open_short])]),
        Event(
            name="平对冲多",
            operate=Operate.LE,
            factors=[
                Factor(name="HX1_1", signals_all=[s_hedge_close]),
                Factor(name="HX1_2", signals_all=[s_main_short]),
                Factor(name="HX1_3", signals_all=[s_main_reverse_to_short]),
                Factor(name="HX1_4", signals_all=[s_main_exit_long]),
                Factor(name="HX1_5", signals_all=[s_main_exit_short]),
            ],
        ),
        Event(
            name="平对冲空",
            operate=Operate.SE,
            factors=[
                Factor(name="HX2_1", signals_all=[s_hedge_close]),
                Factor(name="HX2_2", signals_all=[s_main_long]),
                Factor(name="HX2_3", signals_all=[s_main_reverse_to_long]),
                Factor(name="HX2_4", signals_all=[s_main_exit_long]),
                Factor(name="HX2_5", signals_all=[s_main_exit_short]),
            ],
        ),
    ]


class PenZoneMt5Strategy(CzscStrategyBase):
    @property
    def positions(self) -> List[Position]:
        e = get_events_pen_zone_mt5()
        h = get_events_pen_zone_mt5_hedge()
        return [
            Position(symbol=self.symbol, name="PenZoneMt5V1", opens=[e[0], e[1]], exits=[e[2], e[3]], T0=True),
            Position(symbol=self.symbol, name="PenZoneMt5HedgeV1", opens=[h[0], h[1]], exits=[h[2], h[3]], T0=True),
        ]

    @property
    def signals_config(self):
        return [
            # 主力信号：指定 freq，让框架传入 CZSC 对象
            {"name": pen_zone_signal_mt5_v1, "freq": "15分钟", "di": 1, "log": True},
            # 对冲信号：不指定 freq，让框架传入 CAT/Trader；内部再取 base_freq 的 CZSC
            {"name": pen_zone_hedge_signal_mt5_v1, "di": 1, "base_freq": "15分钟", "pos_name": "PenZoneMt5V1", "hedge_pos_name": "PenZoneMt5HedgeV1", "atr_be_multiplier": 2.0, "log": True},
            # 旧版“2ATR 直接平主力”的信号不再默认启用；对齐 MT5 时由 hedge signal 负责平对冲+解锁
            # {"name": pos_pen_zone_atr_be_mt5_v1, "pos_name": "PenZoneMt5V1", "atr_be_multiplier": 2.0}
        ]
