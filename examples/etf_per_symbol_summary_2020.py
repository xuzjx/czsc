# -*- coding: utf-8 -*-
"""PenZoneA1 ETF 回测：逐品种 2020 年交易统计汇总。

2020年收益% = cum_bp/10000*100（当年总收益，非年化）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from return_metrics import total_return_pct

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_T1_CSV = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "etf_t1_per_symbol_2020.csv")
OUTPUT_T0_CSV = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "etf_t0_per_symbol_2020.csv")

YEAR_START = pd.Timestamp("2020-01-01")
YEAR_END = pd.Timestamp("2020-12-31 23:59:59")

SCENARIOS: List[Tuple[str, str, str]] = [
    ("etf_t1_2020", "PenZoneA1EtfV1", "_bt_pen_zone_a1_stop_etf_all_t1_20200101_20230101_daily"),
    ("etf_t0_2020", "PenZoneA1V1", "_bt_pen_zone_a1_stop_etf_all_20200101_20230101_daily"),
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
    "return_2020_pct": "2020年收益%",
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


def _filter_pairs_2020(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty:
        return dfp
    open_col, close_col = _pairs_time_columns(dfp)
    if not open_col or not close_col:
        return dfp.iloc[0:0]
    open_t = pd.to_datetime(dfp[open_col], errors="coerce")
    close_t = pd.to_datetime(dfp[close_col], errors="coerce")
    mask = (open_t >= YEAR_START) & (open_t <= YEAR_END) & (close_t >= YEAR_START) & (close_t <= YEAR_END)
    return dfp.loc[mask]


def _filter_holds_2020(dfh: pd.DataFrame) -> pd.DataFrame:
    if dfh.empty:
        return dfh
    dt = pd.to_datetime(dfh["dt"], errors="coerce")
    return dfh.loc[(dt >= YEAR_START) & (dt <= YEAR_END)]


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
    dfp20 = _filter_pairs_2020(dfp)

    if dfp20.empty:
        bp = np.array([], dtype=float)
    else:
        bp_col = _pairs_bp_column(dfp20)
        bp = pd.to_numeric(dfp20[bp_col], errors="coerce").fillna(0.0).to_numpy()

    cum_bp = float(bp.sum())
    trades = int(len(bp))
    win_pct = float((bp > 0).mean() * 100.0) if trades else np.nan
    avg_bp = float(bp.mean()) if trades else np.nan
    return_2020_pct = total_return_pct(cum_bp)

    sharpe = np.nan
    if os.path.exists(holds_path):
        dfh = pd.read_parquet(holds_path)
        dfh20 = _filter_holds_2020(dfh)
        if len(dfh20) >= 2:
            pos = dfh20["pos"].astype(float).to_numpy()
            n1b = pd.to_numeric(dfh20["n1b"], errors="coerce").fillna(0.0).to_numpy()
            bar_ret = n1b * pos / 10000.0
            sharpe = _sharpe_from_bar_returns(bar_ret, 1.0)

    return {
        "scenario": scenario,
        "symbol": symbol,
        "cn_name": name_map.get(symbol, ""),
        "trades": trades,
        "win_pct": round(win_pct, 2) if np.isfinite(win_pct) else np.nan,
        "cum_bp": round(cum_bp, 2),
        "avg_bp": round(avg_bp, 2) if np.isfinite(avg_bp) else np.nan,
        "return_2020_pct": round(return_2020_pct, 4),
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
        "cum_bp", "avg_bp", "return_2020_pct", "sharpe",
    ]
    return pd.DataFrame(rows)[col_order]


def _print_top_bottom(df: pd.DataFrame, scenario: str, label: str, n: int = 10) -> None:
    sub = df[df["scenario"] == scenario].copy()
    sub = sub.sort_values("return_2020_pct", ascending=False, na_position="last")
    print(f"\n=== {label} | {len(sub)} symbols ===")
    print(f"\n--- Top {n} by return_2020_pct ---")
    for _, r in sub.head(n).iterrows():
        print(
            f"  {r['symbol']:12s} {(r['cn_name'] or '?'):12s} "
            f"ret%={r['return_2020_pct']:.2f}  cum_bp={r['cum_bp']:.0f}  "
            f"trades={int(r['trades'])}  win%={r['win_pct']:.1f}  sharpe={r['sharpe']}"
        )
    print(f"\n--- Bottom {n} by return_2020_pct ---")
    for _, r in sub.tail(n).iloc[::-1].iterrows():
        print(
            f"  {r['symbol']:12s} {(r['cn_name'] or '?'):12s} "
            f"ret%={r['return_2020_pct']:.2f}  cum_bp={r['cum_bp']:.0f}  "
            f"trades={int(r['trades'])}  win%={r['win_pct']:.1f}  sharpe={r['sharpe']}"
        )


def main():
    df = run_all()
    os.makedirs(os.path.dirname(OUTPUT_T1_CSV), exist_ok=True)

    t1 = df[df["scenario"] == "etf_t1_2020"].sort_values(
        "return_2020_pct", ascending=False, na_position="last"
    )
    t0 = df[df["scenario"] == "etf_t0_2020"].sort_values(
        "return_2020_pct", ascending=False, na_position="last"
    )
    t1.rename(columns=COLUMN_RENAME).to_csv(OUTPUT_T1_CSV, index=False, encoding="utf-8-sig")
    t0.rename(columns=COLUMN_RENAME).to_csv(OUTPUT_T0_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved {len(t1)} rows -> {OUTPUT_T1_CSV}")
    print(f"Saved {len(t0)} rows -> {OUTPUT_T0_CSV}")
    _print_top_bottom(df, "etf_t1_2020", "T+1 2020")
    _print_top_bottom(df, "etf_t0_2020", "T+0 2020")


if __name__ == "__main__":
    main()
