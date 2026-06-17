# -*- coding: utf-8 -*-
"""PenZoneA1 ETF 回测：逐品种交易统计与年化收益汇总。

多区间年化采用线性年化：线性年化% = cum_bp/10000/years*100。
单自然年汇总见 etf_per_symbol_summary_2020.py / etf_per_symbol_summary_2025.py。
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from return_metrics import linear_ann_return_pct

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_CSV = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "etf_per_symbol_summary.csv")
OUTPUT_T1_CSV = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "etf_t1_per_symbol.csv")

SCENARIOS: List[Tuple[str, str, str]] = [
    ("etf_t1_2020_2023", "PenZoneA1EtfV1", "_bt_pen_zone_a1_stop_etf_all_t1_20200101_20230101_daily"),
    ("etf_t0_2020_2023", "PenZoneA1V1", "_bt_pen_zone_a1_stop_etf_all_20200101_20230101_daily"),
    ("etf_t0_2021_2023", "PenZoneA1V1", "_bt_pen_zone_a1_stop_etf_all_20210101_20230101_daily"),
]

ETF_NAME_XLSX = os.path.join(
    os.environ.get("czsc_research_cache", r"D:\CZSC投研数据"),
    "A股场内基金",
    "etfs.xlsx",
)

COLUMN_RENAME = {
    "scenario": "场景",
    "symbol": "代码",
    "cn_name": "名称",
    "trades": "交易次数",
    "win_pct": "胜率%",
    "cum_bp": "累计BP",
    "avg_bp": "均笔BP",
    "ann_pct": "线性年化%",
    "years": "年数",
    "sharpe": "夏普",
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


def _pairs_time_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    open_col, close_col = None, None
    for c in df.columns:
        cs = str(c)
        if "开仓" in cs and "时间" in cs:
            open_col = c
        if "平仓" in cs and "时间" in cs:
            close_col = c
    return open_col, close_col


def _years_from_holds(dfh: pd.DataFrame) -> float:
    sdt = pd.to_datetime(dfh["dt"].iloc[0])
    edt = pd.to_datetime(dfh["dt"].iloc[-1])
    y = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)
    return max(y, 1 / 365.25)


def _years_from_pairs(dfp: pd.DataFrame) -> Optional[float]:
    if dfp.empty:
        return None
    open_col, close_col = _pairs_time_columns(dfp)
    if not open_col or not close_col:
        return None
    sdt = pd.to_datetime(dfp[open_col], errors="coerce").min()
    edt = pd.to_datetime(dfp[close_col], errors="coerce").max()
    if pd.isna(sdt) or pd.isna(edt):
        return None
    y = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)
    return max(y, 1 / 365.25)


def _sharpe_from_bar_returns(returns: np.ndarray, years: float) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 2 or years <= 0:
        return np.nan
    std = np.std(r, ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.nan
    periods_per_year = len(r) / years
    return float(np.mean(r) / std * np.sqrt(periods_per_year))


def analyze_symbol(
    scenario: str,
    symbol: str,
    pos_name: str,
    bt_root: str,
    name_map: Dict[str, str],
) -> dict:
    sym_dir = os.path.join(bt_root, "results", "poss", symbol)
    pairs_path = os.path.join(sym_dir, f"{pos_name}.pairs")
    holds_path = os.path.join(sym_dir, f"{pos_name}.holds")

    dfp = pd.read_parquet(pairs_path)
    if dfp.empty:
        bp = np.array([], dtype=float)
    else:
        bp_col = _pairs_bp_column(dfp)
        bp = pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).to_numpy()

    cum_bp = float(bp.sum())
    trades = int(len(bp))
    win_pct = float((bp > 0).mean() * 100.0) if trades else np.nan
    avg_bp = float(bp.mean()) if trades else np.nan

    years = None
    sharpe = np.nan
    if os.path.exists(holds_path):
        dfh = pd.read_parquet(holds_path)
        if not dfh.empty:
            years = _years_from_holds(dfh)
            pos = dfh["pos"].astype(float).to_numpy()
            n1b = pd.to_numeric(dfh["n1b"], errors="coerce").fillna(0.0).to_numpy()
            bar_ret = n1b * pos / 10000.0
            sharpe = _sharpe_from_bar_returns(bar_ret, years)

    if years is None:
        years = _years_from_pairs(dfp)
    if years is None:
        # fallback from directory tag
        tag = os.path.basename(bt_root)
        parts = tag.split("_")
        for i, p in enumerate(parts):
            if len(p) == 8 and p.isdigit() and i + 1 < len(parts):
                nxt = parts[i + 1]
                if len(nxt) == 8 and nxt.isdigit():
                    sdt = pd.Timestamp(p)
                    edt = pd.Timestamp(nxt)
                    years = max((edt - sdt).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25)
                    break
    if years is None:
        years = 3.0

    ann_pct = linear_ann_return_pct(cum_bp, years)

    return {
        "scenario": scenario,
        "symbol": symbol,
        "cn_name": name_map.get(symbol, ""),
        "trades": trades,
        "win_pct": round(win_pct, 2) if np.isfinite(win_pct) else np.nan,
        "cum_bp": round(cum_bp, 2),
        "avg_bp": round(avg_bp, 2) if np.isfinite(avg_bp) else np.nan,
        "ann_pct": round(ann_pct, 4),
        "years": round(years, 4),
        "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
    }


def run_all() -> pd.DataFrame:
    name_map = _load_etf_names()
    rows = []
    for scenario, pos_name, dir_name in SCENARIOS:
        bt_root = os.path.join(_EXAMPLES_DIR, dir_name)
        poss = os.path.join(bt_root, "results", "poss")
        if not os.path.isdir(poss):
            continue
        for symbol in sorted(os.listdir(poss)):
            sym_dir = os.path.join(poss, symbol)
            pairs_path = os.path.join(sym_dir, f"{pos_name}.pairs")
            if not os.path.isdir(sym_dir) or not os.path.exists(pairs_path):
                continue
            rows.append(analyze_symbol(scenario, symbol, pos_name, bt_root, name_map))
    col_order = [
        "scenario", "symbol", "cn_name", "trades", "win_pct",
        "cum_bp", "avg_bp", "ann_pct", "years", "sharpe",
    ]
    return pd.DataFrame(rows)[col_order]


def _print_top_bottom(df: pd.DataFrame, n: int = 10) -> None:
    sub = df[df["scenario"] == "etf_t1_2020_2023"].copy()
    sub = sub.sort_values("ann_pct", ascending=False, na_position="last")
    print(f"\n=== T+1 (etf_t1_2020_2023) | {len(sub)} symbols ===")
    print(f"\n--- Top {n} by ann_pct ---")
    for _, r in sub.head(n).iterrows():
        print(
            f"  {r['symbol']:12s} {(r['cn_name'] or '?'):12s} "
            f"ann%={r['ann_pct']:.2f}  cum_bp={r['cum_bp']:.0f}  "
            f"trades={int(r['trades'])}  win%={r['win_pct']:.1f}"
        )
    print(f"\n--- Bottom {n} by ann_pct ---")
    for _, r in sub.tail(n).iloc[::-1].iterrows():
        print(
            f"  {r['symbol']:12s} {(r['cn_name'] or '?'):12s} "
            f"ann%={r['ann_pct']:.2f}  cum_bp={r['cum_bp']:.0f}  "
            f"trades={int(r['trades'])}  win%={r['win_pct']:.1f}"
        )


def main():
    df = run_all()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.rename(columns=COLUMN_RENAME).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    t1 = df[df["scenario"] == "etf_t1_2020_2023"].sort_values(
        "ann_pct", ascending=False, na_position="last"
    )
    t1.rename(columns=COLUMN_RENAME).to_csv(OUTPUT_T1_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} rows -> {OUTPUT_CSV}")
    print(f"Saved {len(t1)} rows -> {OUTPUT_T1_CSV}")
    _print_top_bottom(df)


if __name__ == "__main__":
    main()
