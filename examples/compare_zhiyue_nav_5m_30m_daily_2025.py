# -*- coding: utf-8 -*-
"""
指月ETF：5分钟 vs 30分钟 vs 日线 组合 NAV 对比（2025）

对比口径：
- native：各频率使用各自“实际回测产出”的标的集合（即 results/poss 下存在 holds 的标的）
- intersection：三频率均有 holds 的标的交集

输出：
examples/_portfolio_nav_results/zhiyue_nav_compare_5m_30m_daily_2025.csv
以及净值曲线 csv：*_native_nav.csv / *_intersection_nav.csv
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.portfolio_nav_from_holds import NavConfig, analyze_backtest, load_all_holds  # noqa: E402

DAILY_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily")
M30_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_30m")
M5_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_5m")

POS_NAME = "PenZoneA1EtfV1"
OUT_DIR = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results")


def _holds_symbols(bt_root: str, pos_name: str) -> List[str]:
    poss = os.path.join(bt_root, "results", "poss")
    if not os.path.isdir(poss):
        return []
    syms = []
    for s in sorted(os.listdir(poss)):
        p = os.path.join(poss, s, f"{pos_name}.holds")
        if os.path.exists(p):
            syms.append(s)
    return syms


def _analyze_filtered(
    bt_root: str,
    cfg: NavConfig,
    scenario_name: str,
    allow_symbols: Optional[List[str]] = None,
) -> Dict:
    if allow_symbols is None:
        res = analyze_backtest(bt_root, cfg, scenario_name=scenario_name)
        return res

    poss_path = os.path.join(bt_root, "results", "poss")
    holds = load_all_holds(poss_path, cfg.pos_name)
    if holds.empty:
        return {"scenario": scenario_name, "error": "no holds"}
    holds = holds[holds["symbol"].astype(str).isin(set(map(str, allow_symbols)))].copy()
    if holds.empty:
        return {"scenario": scenario_name, "error": "no holds after filter"}

    # 临时复用 analyze_backtest 的计算逻辑：用一个“假 root”方式不方便，这里直接调用核心函数
    # 最简单做法：把过滤后的 holds 写回 analyze_backtest 的流程中（复制少量逻辑，避免改库文件）。
    from examples.portfolio_nav_from_holds import build_portfolio_nav, annualized_return, max_drawdown  # noqa: E402
    import numpy as np  # noqa: E402

    nav_df, _ = build_portfolio_nav(holds, cfg)
    if nav_df.empty:
        return {"scenario": scenario_name, "error": "empty nav"}

    sdt = nav_df["dt"].iloc[0]
    edt = nav_df["dt"].iloc[-1]
    years = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)
    nav_start = cfg.initial_capital
    nav_end = float(nav_df["nav"].iloc[-1])
    ann = annualized_return(nav_start, nav_end, years)

    def _round_ann_pct(rate: float) -> float:
        if not np.isfinite(rate):
            return np.nan
        return round(float(rate) * 100, 2)

    return {
        "scenario": scenario_name,
        "symbols": int(holds["symbol"].nunique()),
        "bars": int(len(nav_df)),
        "start": sdt,
        "end": edt,
        "years": round(float(years), 4),
        "initial_capital": float(cfg.initial_capital),
        "nav_end": round(nav_end, 2),
        "total_return_pct": round((nav_end / nav_start - 1) * 100, 2) if nav_start > 0 else np.nan,
        "ann_return_pct": _round_ann_pct(ann),
        "max_dd_pct": round(max_drawdown(nav_df["nav"]) * 100, 2),
        "nav_df": nav_df,
    }


def _save_nav(nav_df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nav_df = nav_df.copy()
    nav_df["dt"] = pd.to_datetime(nav_df["dt"])
    nav_df[["dt", "nav", "ret"]].to_csv(path, index=False, encoding="utf-8-sig")


def _one_freq(
    tag: str,
    root: str,
    cfg: NavConfig,
    allow_symbols: Optional[List[str]],
) -> Tuple[Dict, Optional[pd.DataFrame]]:
    scenario = f"zhiyue_{tag}"
    res = _analyze_filtered(root, cfg, scenario, allow_symbols=allow_symbols)
    if "error" in res:
        return res, None
    nav_df = res.get("nav_df")
    res = {k: v for k, v in res.items() if k != "nav_df"}
    return res, nav_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = NavConfig(initial_capital=1_000_000.0, pos_name=POS_NAME)

    syms_daily = _holds_symbols(DAILY_ROOT, POS_NAME)
    syms_30m = _holds_symbols(M30_ROOT, POS_NAME)
    syms_5m = _holds_symbols(M5_ROOT, POS_NAME)

    inter = sorted(set(syms_daily) & set(syms_30m) & set(syms_5m))

    rows = []
    for scope, allow in [("native", None), ("intersection", inter)]:
        for tag, root in [("daily", DAILY_ROOT), ("30m", M30_ROOT), ("5m", M5_ROOT)]:
            res, nav_df = _one_freq(f"{tag}_{scope}", root, cfg, allow_symbols=allow)
            if nav_df is not None:
                _save_nav(nav_df, os.path.join(OUT_DIR, f"zhiyue_{tag}_{scope}_nav.csv"))
            row = {
                "scope": scope,
                "freq": tag,
                "symbols": res.get("symbols"),
                "years": res.get("years"),
                "ann_return_pct": res.get("ann_return_pct"),
                "total_return_pct": res.get("total_return_pct"),
                "max_dd_pct": res.get("max_dd_pct"),
                "nav_end": res.get("nav_end"),
                "start": res.get("start"),
                "end": res.get("end"),
                "error": res.get("error", ""),
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "zhiyue_nav_compare_5m_30m_daily_2025.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved -> {out_path}")
    print(f"daily holds symbols={len(syms_daily)} | 30m={len(syms_30m)} | 5m={len(syms_5m)} | intersection={len(inter)}")
    print(f"nav curves -> {OUT_DIR}")


if __name__ == "__main__":
    main()

