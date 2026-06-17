# -*- coding: utf-8 -*-
"""从 DummyBacktest .pairs + holds 导出逐笔开平仓明细（含价格与金额）。

金额计算沿用 portfolio_nav_from_holds：等权分配 initial_capital / N，
ETF 按股数 int(alloc // open_px)，开仓金额=股数*开仓价，平仓金额=股数*平仓价。

用法：
    python examples/export_etf_trades_detail.py
    python examples/export_etf_trades_detail.py --root examples/_bt_pen_zone_a1_stop_etf_all_t1_20250101_20260101_daily
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.portfolio_nav_from_holds import (  # noqa: E402
    NavConfig,
    infer_asset_spec,
    list_holds_files,
    notional,
    simulate_symbol_nav,
)
from examples.pairs_year_end import load_pairs_with_year_end_close  # noqa: E402

DEFAULT_BT_ROOT = os.path.join(
    _EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_all_t1_20250101_20260101_daily"
)
DEFAULT_OUTPUT = os.path.join(
    _EXAMPLES_DIR, "_portfolio_nav_results", "etf_t1_trades_detail_2025.csv"
)
DEFAULT_POS_NAME = "PenZoneA1EtfV1"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0

YEAR_START = pd.Timestamp("2025-01-01")
YEAR_END = pd.Timestamp("2025-12-31 23:59:59")

ETF_NAME_XLSX = os.path.join(
    os.environ.get("czsc_research_cache", r"D:\CZSC投研数据"),
    "A股场内基金",
    "etfs.xlsx",
)

COLUMN_RENAME = {
    "scenario": "场景",
    "symbol": "代码",
    "cn_name": "名称",
    "strategy": "策略标记",
    "direction": "交易方向",
    "open_dt": "开仓时间",
    "close_dt": "平仓时间",
    "open_px": "开仓价格",
    "close_px": "平仓价格",
    "shares": "股数",
    "open_amount": "开仓金额",
    "close_amount": "平仓金额",
    "hold_bars": "持仓K线数",
    "hold_days": "持仓天数",
    "pnl_bp": "盈亏比例",
    "pnl_pct": "盈亏百分比",
    "alloc": "分配资金",
}


def _load_etf_names() -> Dict[str, str]:
    if not os.path.exists(ETF_NAME_XLSX):
        return {}
    try:
        df = pd.read_excel(ETF_NAME_XLSX, usecols=["ts_code", "name"])
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def _pairs_bp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "盈亏" in str(c) and "比例" in str(c):
            return c
    return df.columns[-1]


def _pairs_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    for c in df.columns:
        if keyword in str(c):
            return c
    return None


def _filter_pairs_year(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty:
        return dfp
    open_col = _pairs_col(dfp, "开仓时间")
    close_col = _pairs_col(dfp, "平仓时间")
    if not open_col or not close_col:
        return dfp.iloc[0:0]
    open_t = pd.to_datetime(dfp[open_col], errors="coerce")
    close_t = pd.to_datetime(dfp[close_col], errors="coerce")
    mask = (open_t >= YEAR_START) & (open_t <= YEAR_END) & (close_t >= YEAR_START) & (close_t <= YEAR_END)
    return dfp.loc[mask].copy()


def _symbol_size_at_open(
    holds_path: str,
    alloc: float,
    cfg: NavConfig,
    spec,
) -> pd.Series:
    """开仓 bar 的 dt -> 股数/手数。"""
    dfh = pd.read_parquet(holds_path)
    if dfh.empty:
        return pd.Series(dtype=int)
    nav = simulate_symbol_nav(dfh, alloc, cfg, spec)
    merged = dfh[["dt"]].merge(nav[["dt", "size"]], on="dt", how="left")
    pos = dfh["pos"].astype(int).to_numpy()
    size = merged["size"].fillna(0).astype(int).to_numpy()
    # 仅保留 pos!=0 且有仓位的 bar（开仓/持仓期间）
    mask = (pos != 0) & (size > 0)
    out = pd.Series(size[mask], index=pd.to_datetime(dfh["dt"].values)[mask])
    return out


def export_trades_detail(
    bt_root: str,
    pos_name: str = DEFAULT_POS_NAME,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    scenario: str = "etf_t1_2025",
    year_filter: bool = True,
) -> pd.DataFrame:
    poss_path = os.path.join(bt_root, "results", "poss")
    if not os.path.isdir(poss_path):
        raise FileNotFoundError(f"未找到 poss 目录: {poss_path}")

    symbols = [s for s, _ in list_holds_files(poss_path, pos_name)]
    if not symbols:
        raise FileNotFoundError(f"未找到 {pos_name}.holds 文件")

    n = len(symbols)
    alloc = initial_capital / n
    cfg = NavConfig(initial_capital=initial_capital, pos_name=pos_name)
    name_map = _load_etf_names()

    rows: List[dict] = []
    for symbol in symbols:
        sym_dir = os.path.join(poss_path, symbol)
        pairs_path = os.path.join(sym_dir, f"{pos_name}.pairs")
        holds_path = os.path.join(sym_dir, f"{pos_name}.holds")
        if not os.path.exists(pairs_path):
            continue

        dfp, _ = load_pairs_with_year_end_close(pairs_path, holds_path, YEAR_END, pos_name)
        if year_filter:
            dfp = _filter_pairs_year(dfp)
        if dfp.empty:
            continue

        spec = infer_asset_spec(symbol)
        size_map = _symbol_size_at_open(holds_path, alloc, cfg, spec)

        sym_col = _pairs_col(dfp, "标的代码") or dfp.columns[0]
        strat_col = _pairs_col(dfp, "策略")
        dir_col = _pairs_col(dfp, "交易方向")
        open_col = _pairs_col(dfp, "开仓时间")
        close_col = _pairs_col(dfp, "平仓时间")
        open_px_col = _pairs_col(dfp, "开仓价格")
        close_px_col = _pairs_col(dfp, "平仓价格")
        bars_col = _pairs_col(dfp, "持仓K线")
        days_col = _pairs_col(dfp, "持仓天数")
        bp_col = _pairs_bp_column(dfp)

        for _, r in dfp.iterrows():
            open_dt = pd.Timestamp(r[open_col])
            open_px = float(r[open_px_col])
            close_px = float(r[close_px_col])
            shares = int(size_map.get(open_dt, 0))
            if shares <= 0 and open_px > 0:
                shares = int(alloc // open_px) if spec.asset_type == "stock" else 0

            rows.append(
                {
                    "scenario": scenario,
                    "symbol": str(r[sym_col]),
                    "cn_name": name_map.get(symbol, ""),
                    "strategy": str(r[strat_col]) if strat_col else pos_name,
                    "direction": str(r[dir_col]) if dir_col else "",
                    "open_dt": open_dt,
                    "close_dt": pd.Timestamp(r[close_col]),
                    "open_px": round(open_px, 4),
                    "close_px": round(close_px, 4),
                    "shares": shares,
                    "open_amount": round(notional(shares, open_px, spec), 2),
                    "close_amount": round(notional(shares, close_px, spec), 2),
                    "hold_bars": int(r[bars_col]) if bars_col else np.nan,
                    "hold_days": round(float(r[days_col]), 2) if days_col else np.nan,
                    "pnl_bp": round(float(r[bp_col]), 2),
                    "pnl_pct": round(float(r[bp_col]) / 100.0, 4),
                    "alloc": round(alloc, 2),
                }
            )

    col_order = list(COLUMN_RENAME.keys())
    return pd.DataFrame(rows)[col_order] if rows else pd.DataFrame(columns=col_order)


def main():
    parser = argparse.ArgumentParser(description="导出 ETF 逐笔开平仓明细")
    parser.add_argument("--root", default=DEFAULT_BT_ROOT, help="回测根目录")
    parser.add_argument("--pos-name", default=DEFAULT_POS_NAME)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--scenario", default="etf_t1_2025")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-year-filter", action="store_true", help="不过滤 2025 年")
    args = parser.parse_args()

    df = export_trades_detail(
        bt_root=args.root,
        pos_name=args.pos_name,
        initial_capital=args.initial_capital,
        scenario=args.scenario,
        year_filter=not args.no_year_filter,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.rename(columns=COLUMN_RENAME).to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} trades -> {args.output}")
    if not df.empty:
        show = df.rename(columns=COLUMN_RENAME).head(5)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 240)
        print("\nSample rows:")
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
