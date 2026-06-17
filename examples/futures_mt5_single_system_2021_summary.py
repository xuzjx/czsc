# -*- coding: utf-8 -*-
"""
PenZoneSingleSystemMt5Strategy 期货回测（2021）汇总：
- 逐品种：年化收益、夏普、交易次数、胜率等
- 组合：截面等权 NAV（按持仓 bar 收益等权）

默认读取：
examples/_bt_pen_zone_single_system_futures_mt5_single_system_20210101_20220101_15m/results/poss

用法：
python examples/futures_mt5_single_system_2021_summary.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))

SDT = pd.Timestamp("2021-01-01")
EDT = pd.Timestamp("2022-01-01")
YEARS = (EDT - SDT).total_seconds() / (365.25 * 24 * 3600)

_BT_ROOT_CANDIDATES = [
    # 推荐：本次任务的标准输出目录
    os.path.join(EXAMPLES_DIR, "_bt_pen_zone_single_system_futures_mt5_single_system_20210101_20220101_15m", "results"),
    # 兼容：仓库里已存在的 2021 小样本回测目录（仅跑了少量品种时也能先出汇总）
    os.path.join(EXAMPLES_DIR, "_bt_pen_zone_single_system_futures_2021", "results_20210101_20211231"),
]


def _pick_bt_root() -> str:
    for p in _BT_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(p, "poss")):
            return p
    # 默认返回第一个，报错时提示清晰的路径
    return _BT_ROOT_CANDIDATES[0]

POS_MAIN = "PenZoneSingleSystemMt5V1"
POS_HEDGE = "PenZoneSingleSystemMt5HedgeV1"

OUT_DIR = os.path.join(EXAMPLES_DIR, "_portfolio_nav_results")
OUT_PER_SYMBOL = os.path.join(OUT_DIR, "futures_mt5_single_system_2021_per_symbol.csv")
OUT_NAV = os.path.join(OUT_DIR, "futures_mt5_single_system_2021_nav.csv")


def _pairs_bp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "盈亏" in str(c) and "比例" in str(c):
            return c
    return df.columns[-1]


def _sharpe_from_bar_returns(returns: np.ndarray, years: float) -> Tuple[float, str]:
    r = returns[np.isfinite(returns)]
    if len(r) < 2 or years <= 0:
        return np.nan, "holds_insufficient"
    std = np.std(r, ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.nan, "holds_zero_std"
    periods_per_year = len(r) / years
    sharpe = float(np.mean(r) / std * np.sqrt(periods_per_year))
    return sharpe, "holds_bar"


def _sharpe_from_trades(trade_returns: np.ndarray, years: float) -> Tuple[float, str]:
    r = trade_returns[np.isfinite(trade_returns)]
    if len(r) < 2 or years <= 0:
        return np.nan, "trades_insufficient"
    std = np.std(r, ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.nan, "trades_zero_std"
    trades_per_year = len(r) / years
    sharpe = float(np.mean(r) / std * np.sqrt(trades_per_year))
    return sharpe, "trades"


def _load_pairs(poss_dir: str, symbol: str, pos_name: str) -> Optional[pd.DataFrame]:
    fp = os.path.join(poss_dir, symbol, f"{pos_name}.pairs")
    if not os.path.exists(fp):
        return None
    return pd.read_parquet(fp)


def _load_holds(poss_dir: str, symbol: str, pos_name: str) -> Optional[pd.DataFrame]:
    fh = os.path.join(poss_dir, symbol, f"{pos_name}.holds")
    if not os.path.exists(fh):
        return None
    return pd.read_parquet(fh)


def analyze_symbol(poss_dir: str, symbol: str, pos_name: str) -> dict:
    dfp = _load_pairs(poss_dir, symbol, pos_name)
    if dfp is None or dfp.empty:
        bp = np.array([], dtype=float)
    else:
        bp_col = _pairs_bp_column(dfp)
        bp = pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).to_numpy()

    cum_bp = float(bp.sum())
    trades = int(len(bp))
    win_rate = float((bp > 0).mean() * 100.0) if trades else np.nan
    avg_bp = float(bp.mean()) if trades else np.nan

    dfh = _load_holds(poss_dir, symbol, pos_name)
    sharpe = np.nan
    sharpe_method = "none"
    n_bars = 0
    if dfh is not None and not dfh.empty:
        n_bars = len(dfh)
        pos = pd.to_numeric(dfh["pos"], errors="coerce").fillna(0.0).to_numpy()
        n1b = pd.to_numeric(dfh["n1b"], errors="coerce").fillna(0.0).to_numpy()
        bar_ret = n1b * pos / 10000.0
        sharpe, sharpe_method = _sharpe_from_bar_returns(bar_ret, YEARS)

    if not np.isfinite(sharpe):
        trade_ret = bp / 10000.0
        sharpe, sharpe_method = _sharpe_from_trades(trade_ret, YEARS)

    ann_linear_pct = (cum_bp / 10000.0 / max(YEARS, 1 / 365.25)) * 100.0
    ann_compound_pct = (np.power(1 + cum_bp / 10000.0, 1 / max(YEARS, 1 / 365.25)) - 1) * 100.0

    return {
        "symbol": symbol,
        "pos_name": pos_name,
        "cum_bp": round(cum_bp, 2),
        "trades": trades,
        "win_rate_pct": round(win_rate, 2) if np.isfinite(win_rate) else np.nan,
        "avg_bp": round(avg_bp, 2) if np.isfinite(avg_bp) else np.nan,
        "n_bars": n_bars,
        "ann_return_linear_pct": round(float(ann_linear_pct), 4),
        "ann_return_compound_pct": round(float(ann_compound_pct), 4),
        "sharpe": round(float(sharpe), 4) if np.isfinite(sharpe) else np.nan,
        "sharpe_method": sharpe_method,
    }


def compute_equal_weight_nav(poss_dir: str, pos_name: str) -> pd.DataFrame:
    """按截面等权：每根bar等权持仓收益 = mean(n1b*pos/10000)（仅统计 pos!=0 的品种）。"""
    symbols = [d for d in os.listdir(poss_dir) if os.path.isdir(os.path.join(poss_dir, d))]
    rows = []
    for symbol in symbols:
        dfh = _load_holds(poss_dir, symbol, pos_name)
        if dfh is None or dfh.empty:
            continue
        dfh = dfh.copy()
        dfh["dt"] = pd.to_datetime(dfh["dt"])
        dfh["pos"] = pd.to_numeric(dfh["pos"], errors="coerce").fillna(0.0)
        dfh["n1b"] = pd.to_numeric(dfh["n1b"], errors="coerce").fillna(0.0)
        dfh = dfh[(dfh["dt"] >= SDT) & (dfh["dt"] < EDT)]
        dfh = dfh[dfh["pos"] != 0]
        if dfh.empty:
            continue
        dfh["ret"] = dfh["n1b"] * dfh["pos"] / 10000.0
        rows.append(dfh[["dt", "ret"]])

    if not rows:
        return pd.DataFrame(columns=["dt", "ret", "nav"])

    allh = pd.concat(rows, ignore_index=True)
    cross = allh.groupby("dt")["ret"].mean().sort_index()
    nav = (1.0 + cross).cumprod()
    out = pd.DataFrame({"dt": cross.index, "ret": cross.values, "nav": nav.values})
    return out


def main():
    bt_root = _pick_bt_root()
    poss_dir = os.path.join(bt_root, "poss")
    if not os.path.isdir(poss_dir):
        raise RuntimeError(f"Backtest results not found: {poss_dir}")

    symbols = [d for d in os.listdir(poss_dir) if os.path.isdir(os.path.join(poss_dir, d))]
    if not symbols:
        raise RuntimeError(f"poss 目录为空：{poss_dir}")

    rows = []
    for symbol in sorted(symbols):
        rows.append(analyze_symbol(poss_dir, symbol, POS_MAIN))
        rows.append(analyze_symbol(poss_dir, symbol, POS_HEDGE))

    df = pd.DataFrame(rows).sort_values(["pos_name", "sharpe"], ascending=[True, False], na_position="last")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PER_SYMBOL, index=False, encoding="utf-8-sig")

    nav = compute_equal_weight_nav(poss_dir, POS_MAIN)
    nav.to_csv(OUT_NAV, index=False, encoding="utf-8-sig")

    print(f"Backtest root: {bt_root}")
    print(f"Saved per-symbol summary -> {OUT_PER_SYMBOL} ({len(df)} rows)")
    print(f"Saved equal-weight NAV    -> {OUT_NAV} ({len(nav)} bars)")


if __name__ == "__main__":
    main()

