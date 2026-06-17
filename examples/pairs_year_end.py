# -*- coding: utf-8 -*-
"""年末仍持仓时，按末 bar 收盘价补合成平仓 pair（用于统计/导出，无需重跑回测）。"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd


def _pairs_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    for c in df.columns:
        if keyword in str(c):
            return c
    return None


def _last_open_in_holds(dfh: pd.DataFrame, year_end: pd.Timestamp) -> Optional[pd.Series]:
    dfh = dfh.copy()
    dfh["dt"] = pd.to_datetime(dfh["dt"], errors="coerce")
    dfh_y = dfh[dfh["dt"] <= year_end].sort_values("dt")
    if dfh_y.empty or int(dfh_y.iloc[-1]["pos"]) == 0:
        return None
    prev = dfh_y["pos"].shift(1).fillna(0)
    opens = dfh_y[(dfh_y["pos"] != 0) & (prev == 0)]
    if opens.empty:
        return None
    return opens.iloc[-1]


def _open_already_closed(dfp: pd.DataFrame, open_dt: pd.Timestamp) -> bool:
    if dfp.empty:
        return False
    open_col = _pairs_col(dfp, "开仓时间")
    if not open_col:
        return False
    open_t = pd.to_datetime(dfp[open_col], errors="coerce")
    return bool((open_t == open_dt).any())


def maybe_append_year_end_close_pair(
    dfp: pd.DataFrame,
    dfh: pd.DataFrame,
    year_end: pd.Timestamp,
    pos_name: str = "",
    desc: str = "年末强平",
) -> pd.DataFrame:
    """若 year_end 前仍持仓且 pairs 中无对应平仓，追加合成 pair。"""
    open_row = _last_open_in_holds(dfh, year_end)
    if open_row is None:
        return dfp

    open_dt = pd.Timestamp(open_row["dt"])
    if _open_already_closed(dfp, open_dt):
        return dfp

    dfh = dfh.copy()
    dfh["dt"] = pd.to_datetime(dfh["dt"], errors="coerce")
    dfh_y = dfh[dfh["dt"] <= year_end].sort_values("dt")
    close_row = dfh_y.iloc[-1]

    open_px = float(open_row["price"])
    close_px = float(close_row["price"])
    close_dt = pd.Timestamp(close_row["dt"])
    last_pos = int(close_row["pos"])
    symbol = str(close_row.get("symbol", dfp.iloc[0][_pairs_col(dfp, "标的代码")] if not dfp.empty and _pairs_col(dfp, "标的代码") else ""))

    if last_pos == 1:
        direction = "多头"
        pnl_bp = round((close_px / open_px - 1) * 10000, 2)
        event_seq = f"开多@{desc} -> 平多@{desc}"
    else:
        direction = "空头"
        pnl_bp = round((1 - close_px / open_px) * 10000, 2)
        event_seq = f"开空@{desc} -> 平空@{desc}"

    hold_slice = dfh_y[(dfh_y["dt"] >= open_dt) & (dfh_y["dt"] <= close_dt)]
    hold_bars = max(len(hold_slice) - 1, 0)
    hold_days = (close_dt - open_dt).total_seconds() / (24 * 3600)

    new_pair = {
        "标的代码": symbol,
        "策略标记": pos_name or (dfp.iloc[0][_pairs_col(dfp, "策略")] if not dfp.empty and _pairs_col(dfp, "策略") else ""),
        "交易方向": direction,
        "开仓时间": open_dt,
        "平仓时间": close_dt,
        "开仓价格": open_px,
        "平仓价格": close_px,
        "持仓K线数": hold_bars,
        "事件序列": event_seq,
        "持仓天数": hold_days,
        "盈亏比例": pnl_bp,
    }
    if dfp.empty:
        return pd.DataFrame([new_pair])
    return pd.concat([dfp, pd.DataFrame([new_pair])], ignore_index=True)


def load_pairs_with_year_end_close(
    pairs_path: str,
    holds_path: str,
    year_end: pd.Timestamp,
    pos_name: str = "",
) -> Tuple[pd.DataFrame, bool]:
    """读取 pairs，必要时按 holds 补年末强平 pair。返回 (dfp, appended)。"""
    dfp = pd.read_parquet(pairs_path)
    if not holds_path or not os.path.exists(holds_path):
        return dfp, False
    dfh = pd.read_parquet(holds_path)
    before = len(dfp)
    dfp = maybe_append_year_end_close_pair(dfp, dfh, year_end, pos_name=pos_name)
    return dfp, len(dfp) > before
