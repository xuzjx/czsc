# -*- coding: utf-8 -*-
"""
指月 ETF：组合净值年化对比（2025）- 日线 vs 30m

目标：
- 从 DummyBacktest 输出的 `.holds` 构建等权组合 NAV（按标的等额资金分配）
- 两种宇宙对比：
  A) 交集宇宙：daily 与 30m 都有 holds 的标的
  B) 原生宇宙：各自目录下所有有 holds 的标的

输出：
- examples/_portfolio_nav_results/zhiyue_nav_compare_daily_vs_30m_2025.csv
- examples/_portfolio_nav_results/zhiyue_nav_daily_2025.csv
- examples/_portfolio_nav_results/zhiyue_nav_30m_2025.csv
- （可选）交集宇宙净值曲线：
  zhiyue_nav_daily_intersection_2025.csv / zhiyue_nav_30m_intersection_2025.csv
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.portfolio_nav_from_holds import (  # noqa: E402
    NavConfig,
    annualized_return,
    build_portfolio_nav,
    max_drawdown,
)


DAILY_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily")
M30_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_30m")
POS_NAME = "PenZoneA1EtfV1"


@dataclass
class NavStats:
    universe: str
    freq: str
    symbols: int
    start: str
    end: str
    total_return_pct: float
    ann_return_pct: float
    max_dd_pct: float
    nav_start: float
    nav_end: float
    bars: int
    nav_curve_csv: str


def _list_symbols_with_holds(bt_root: str, pos_name: str) -> List[str]:
    poss = os.path.join(bt_root, "results", "poss")
    if not os.path.isdir(poss):
        return []
    out = []
    for sym in sorted(os.listdir(poss)):
        f = os.path.join(poss, sym, f"{pos_name}.holds")
        if os.path.exists(f):
            out.append(sym)
    return out


def _load_holds_for_symbols(bt_root: str, pos_name: str, symbols: List[str]) -> pd.DataFrame:
    poss = os.path.join(bt_root, "results", "poss")
    rows = []
    for sym in symbols:
        f = os.path.join(poss, sym, f"{pos_name}.holds")
        if not os.path.exists(f):
            continue
        dfh = pd.read_parquet(f)
        if "symbol" not in dfh.columns:
            dfh["symbol"] = sym
        rows.append(dfh)
    if not rows:
        return pd.DataFrame(columns=["dt", "pos", "price", "n1b", "symbol"])
    out = pd.concat(rows, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"])
    return out.sort_values(["symbol", "dt"]).reset_index(drop=True)


def _compute_nav_stats(
    bt_root: str,
    freq: str,
    universe: str,
    symbols: List[str],
    cfg: NavConfig,
    out_curve_path: str,
) -> Tuple[NavStats, pd.DataFrame]:
    holds = _load_holds_for_symbols(bt_root, cfg.pos_name, symbols)
    if holds.empty:
        raise SystemExit(f"[{freq}][{universe}] no holds loaded")

    nav_df, _ = build_portfolio_nav(holds, cfg)
    nav_df = nav_df.sort_values("dt").reset_index(drop=True)
    nav_2025 = nav_df[(nav_df["dt"] >= "2025-01-01") & (nav_df["dt"] <= "2025-12-31")].copy()
    if nav_2025.empty:
        raise SystemExit(f"[{freq}][{universe}] no nav rows in 2025")

    sdt = pd.to_datetime(nav_2025["dt"].iloc[0])
    edt = pd.to_datetime(nav_2025["dt"].iloc[-1])
    years = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)

    nav_start = float(nav_2025["nav"].iloc[0])
    nav_end = float(nav_2025["nav"].iloc[-1])
    total_ret = nav_end / nav_start - 1.0 if nav_start > 0 else np.nan
    ann = annualized_return(nav_start, nav_end, years)
    mdd = max_drawdown(nav_2025["nav"])

    os.makedirs(os.path.dirname(out_curve_path), exist_ok=True)
    nav_2025[["dt", "nav", "ret"]].to_csv(out_curve_path, index=False, encoding="utf-8-sig")

    stats = NavStats(
        universe=universe,
        freq=freq,
        symbols=len(symbols),
        start=str(sdt.date()),
        end=str(edt.date()),
        total_return_pct=round(total_ret * 100, 2) if np.isfinite(total_ret) else np.nan,
        ann_return_pct=round(float(ann) * 100, 2) if np.isfinite(ann) else np.nan,
        max_dd_pct=round(float(mdd) * 100, 2) if np.isfinite(mdd) else np.nan,
        nav_start=round(nav_start, 2),
        nav_end=round(nav_end, 2),
        bars=len(nav_2025),
        nav_curve_csv=os.path.basename(out_curve_path),
    )
    return stats, nav_2025


def main():
    out_dir = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results")
    os.makedirs(out_dir, exist_ok=True)

    daily_syms = _list_symbols_with_holds(DAILY_ROOT, POS_NAME)
    m30_syms = _list_symbols_with_holds(M30_ROOT, POS_NAME)
    inter_syms = sorted(set(daily_syms).intersection(set(m30_syms)))

    cfg = NavConfig(
        initial_capital=1_000_000.0,
        fee_rate=0.0003,
        slippage=0.0001,
        pos_name=POS_NAME,
        futures_fee_rate=0.00005,
        futures_slippage=0.00002,
    )

    rows: List[Dict] = []

    # Native universes
    s1, _ = _compute_nav_stats(
        DAILY_ROOT,
        freq="daily",
        universe="native",
        symbols=daily_syms,
        cfg=cfg,
        out_curve_path=os.path.join(out_dir, "zhiyue_nav_daily_2025.csv"),
    )
    rows.append(asdict(s1))

    s2, _ = _compute_nav_stats(
        M30_ROOT,
        freq="30m",
        universe="native",
        symbols=m30_syms,
        cfg=cfg,
        out_curve_path=os.path.join(out_dir, "zhiyue_nav_30m_2025.csv"),
    )
    rows.append(asdict(s2))

    # Intersection universe
    s3, _ = _compute_nav_stats(
        DAILY_ROOT,
        freq="daily",
        universe="intersection",
        symbols=inter_syms,
        cfg=cfg,
        out_curve_path=os.path.join(out_dir, "zhiyue_nav_daily_intersection_2025.csv"),
    )
    rows.append(asdict(s3))

    s4, _ = _compute_nav_stats(
        M30_ROOT,
        freq="30m",
        universe="intersection",
        symbols=inter_syms,
        cfg=cfg,
        out_curve_path=os.path.join(out_dir, "zhiyue_nav_30m_intersection_2025.csv"),
    )
    rows.append(asdict(s4))

    out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "zhiyue_nav_compare_daily_vs_30m_2025.csv")
    out = out[
        [
            "universe",
            "freq",
            "symbols",
            "start",
            "end",
            "total_return_pct",
            "ann_return_pct",
            "max_dd_pct",
            "nav_start",
            "nav_end",
            "bars",
            "nav_curve_csv",
        ]
    ].sort_values(["universe", "freq"])
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved -> {out_path}")
    print(f"daily_symbols={len(daily_syms)} | 30m_symbols={len(m30_syms)} | intersection={len(inter_syms)}")


if __name__ == "__main__":
    main()

