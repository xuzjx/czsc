# -*- coding: utf-8 -*-
"""PenZoneA1 期货回测：逐品种年化收益与夏普比率汇总。

读取 examples/_bt_pen_zone_a1_stop_futures_20210101_20230101_* 下
results/poss/{symbol}/PenZoneA1V1.pairs 与 .holds，输出完整 CSV。
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from return_metrics import compound_ann_return_pct, linear_ann_return_pct

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))

# czsc.connectors.tq_connector.future_name_map（避免 tqsdk 依赖）
FUTURE_NAME_MAP = {
    "AO": "氧化铝", "PX": "对二甲苯", "EC": "欧线集运", "LC": "碳酸锂", "PG": "LPG",
    "EB": "苯乙烯", "CS": "玉米淀粉", "C": "玉米", "V": "PVC", "J": "焦炭", "BB": "胶合板",
    "M": "豆粕", "A": "豆一", "PP": "聚丙烯", "P": "棕榈油", "FB": "纤维板", "B": "豆二",
    "JD": "鸡蛋", "JM": "焦煤", "L": "塑料", "I": "铁矿石", "Y": "豆油", "RR": "粳米",
    "EG": "乙二醇", "LH": "生猪", "CJ": "红枣", "UR": "尿素", "TA": "PTA", "OI": "菜油",
    "MA": "甲醇", "RS": "菜籽", "ZC": "动力煤", "LR": "晚籼稻", "PM": "普麦", "SR": "白糖",
    "RI": "早籼稻", "SF": "硅铁", "WH": "强麦", "JR": "粳稻", "SM": "锰硅", "FG": "玻璃",
    "CF": "棉花", "RM": "菜粕", "PF": "短纤", "AP": "苹果", "CY": "棉纱", "SA": "纯碱",
    "PK": "花生", "SS": "不锈钢", "AL": "沪铝", "CU": "沪铜", "ZN": "沪锌", "AG": "白银",
    "RB": "螺纹钢", "SN": "沪锡", "NI": "沪镍", "WR": "线材", "FU": "燃油", "AU": "黄金",
    "PB": "沪铅", "RU": "橡胶", "BR": "合成橡胶", "HC": "热轧卷板", "BU": "沥青", "SP": "纸浆",
    "NR": "20号胶", "SC": "原油", "LU": "低硫燃料油", "BC": "国际铜", "SI": "工业硅",
    "IF": "沪深300", "IH": "上证50", "IC": "中证500", "IM": "中证1000",
    "T": "10年国债", "TF": "5年国债", "TS": "2年国债",
}

POS_NAME = "PenZoneA1V1"
BT_GLOB_PREFIX = "_bt_pen_zone_a1_stop_futures_20210101_20230101_"
OUTPUT_CSV = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "futures_per_symbol_ann_sharpe.csv")

# 回测名义区间（用于无 holds 时的 fallback）
NOMINAL_SDT = pd.Timestamp("2021-01-01")
NOMINAL_EDT = pd.Timestamp("2023-01-01")
NOMINAL_YEARS = (NOMINAL_EDT - NOMINAL_SDT).total_seconds() / (365.25 * 24 * 3600)

TF_LABEL = {
    "1m": "1分钟",
    "5m": "5分钟",
    "15m": "15分钟",
    "120m": "120分钟",
    "daily": "日线",
}


def _parse_futures_product(symbol: str) -> str:
    body = symbol[:-4] if symbol.endswith("9001") else symbol
    return re.sub(r"^[A-Z]{2}", "", body)


def _cn_name(symbol: str) -> str:
    prod = _parse_futures_product(symbol)
    return FUTURE_NAME_MAP.get(prod.upper(), FUTURE_NAME_MAP.get(prod, ""))


def _pairs_bp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "盈亏" in str(c) and "比例" in str(c):
            return c
    return df.columns[-1]


def _years_from_holds(dfh: pd.DataFrame) -> float:
    if dfh is None or dfh.empty:
        return NOMINAL_YEARS
    sdt = pd.to_datetime(dfh["dt"].iloc[0])
    edt = pd.to_datetime(dfh["dt"].iloc[-1])
    y = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)
    return max(y, 1 / 365.25)


def _sharpe_from_bar_returns(returns: np.ndarray, years: float) -> Tuple[float, str]:
    """holds 逐 bar 收益序列年化夏普。"""
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


def discover_futures_bt_dirs(examples_dir: str = _EXAMPLES_DIR) -> List[Tuple[str, str]]:
    """返回 [(timeframe_key, abs_path), ...]"""
    out = []
    for name in sorted(os.listdir(examples_dir)):
        if not name.startswith(BT_GLOB_PREFIX):
            continue
        tf = name[len(BT_GLOB_PREFIX):]
        path = os.path.join(examples_dir, name)
        poss = os.path.join(path, "results", "poss")
        if os.path.isdir(poss):
            out.append((tf, path))
    return out


def analyze_symbol(
    symbol: str,
    pairs_path: str,
    holds_path: Optional[str],
    timeframe: str,
) -> dict:
    dfp = pd.read_parquet(pairs_path)
    if dfp.empty:
        bp = np.array([], dtype=float)
    else:
        bp_col = _pairs_bp_column(dfp)
        bp = pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).to_numpy()
    cum_bp = float(bp.sum())
    trades = int(len(bp))
    win_rate = float((bp > 0).mean() * 100.0) if trades else np.nan
    avg_bp = float(bp.mean()) if trades else np.nan

    dfh = None
    years = NOMINAL_YEARS
    n_bars = 0
    sharpe = np.nan
    sharpe_method = "none"

    if holds_path and os.path.exists(holds_path):
        dfh = pd.read_parquet(holds_path)
        if not dfh.empty:
            n_bars = len(dfh)
            years = _years_from_holds(dfh)
            pos = dfh["pos"].astype(float).to_numpy()
            n1b = pd.to_numeric(dfh["n1b"], errors="coerce").fillna(0.0).to_numpy()
            bar_ret = n1b * pos / 10000.0
            sharpe, sharpe_method = _sharpe_from_bar_returns(bar_ret, years)

    if not np.isfinite(sharpe):
        trade_ret = bp / 10000.0
        sharpe, sharpe_method = _sharpe_from_trades(trade_ret, years)

    prod = _parse_futures_product(symbol)
    return {
        "timeframe": timeframe,
        "timeframe_cn": TF_LABEL.get(timeframe, timeframe),
        "symbol": symbol,
        "product": prod,
        "cn_name": _cn_name(symbol),
        "cum_bp": round(cum_bp, 2),
        "trades": trades,
        "win_rate_pct": round(win_rate, 2) if np.isfinite(win_rate) else np.nan,
        "avg_bp": round(avg_bp, 2) if np.isfinite(avg_bp) else np.nan,
        "years": round(years, 4),
        "n_bars": n_bars,
        "ann_return_linear_pct": round(linear_ann_return_pct(cum_bp, years), 4),
        "ann_return_compound_pct": round(compound_ann_return_pct(cum_bp, years), 4),
        "sharpe": round(sharpe, 4) if np.isfinite(sharpe) else np.nan,
        "sharpe_method": sharpe_method,
    }


def run_all() -> pd.DataFrame:
    rows = []
    for tf, bt_root in discover_futures_bt_dirs():
        poss = os.path.join(bt_root, "results", "poss")
        for symbol in sorted(os.listdir(poss)):
            sym_dir = os.path.join(poss, symbol)
            if not os.path.isdir(sym_dir):
                continue
            pairs_path = os.path.join(sym_dir, f"{POS_NAME}.pairs")
            if not os.path.exists(pairs_path):
                continue
            holds_path = os.path.join(sym_dir, f"{POS_NAME}.holds")
            rows.append(analyze_symbol(symbol, pairs_path, holds_path, tf))
    df = pd.DataFrame(rows)
    col_order = [
        "timeframe", "timeframe_cn", "symbol", "product", "cn_name",
        "cum_bp", "trades", "win_rate_pct", "avg_bp", "years", "n_bars",
        "ann_return_linear_pct", "ann_return_compound_pct",
        "sharpe", "sharpe_method",
    ]
    return df[col_order].sort_values(["timeframe", "sharpe"], ascending=[True, False], na_position="last")


def print_top_bottom(df: pd.DataFrame, tf: str, n: int = 5) -> None:
    sub = df[df["timeframe"] == tf].copy()
    if sub.empty:
        print(f"\n=== {tf}: 无数据 ===")
        return
    print(f"\n=== {TF_LABEL.get(tf, tf)} ({tf}) | {len(sub)} 品种 ===")
    for metric, label in [("sharpe", "Sharpe"), ("ann_return_linear_pct", "线性年化%")]:
        valid = sub[sub[metric].notna()].sort_values(metric, ascending=False)
        print(f"\n--- Top {n} by {label} ---")
        for _, r in valid.head(n).iterrows():
            print(
                f"  {r['symbol']:12s} {r['cn_name'] or '?':8s} "
                f"{label}={r[metric]:.4f}  cum_bp={r['cum_bp']:.0f}  trades={int(r['trades'])}"
            )
        print(f"--- Bottom {n} by {label} ---")
        for _, r in valid.tail(n).iloc[::-1].iterrows():
            print(
                f"  {r['symbol']:12s} {r['cn_name'] or '?':8s} "
                f"{label}={r[metric]:.4f}  cum_bp={r['cum_bp']:.0f}  trades={int(r['trades'])}"
            )


def main():
    df = run_all()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows -> {OUTPUT_CSV}")
    print(f"\nSharpe methodology: primary=holds bar returns (n1b*pos/10000), "
          f"annualized Sharpe=mean/std*sqrt(bars/years); fallback=per-trade bp/10000.")
    print(f"Ann return: linear = cum_bp/10000/years*100; compound also in ann_return_compound_pct.")
    for tf in ("5m", "15m"):
        print_top_bottom(df, tf)
    print("\n--- All timeframes summary (median sharpe / median ann%) ---")
    for tf in sorted(df["timeframe"].unique()):
        sub = df[df["timeframe"] == tf]
        print(
            f"  {tf:6s}: n={len(sub):2d}  "
            f"med_sharpe={sub['sharpe'].median():.3f}  "
            f"med_ann%={sub['ann_return_linear_pct'].median():.2f}  "
            f"pos_sharpe={(sub['sharpe']>0).mean()*100:.0f}%  "
            f"pos_ann={(sub['ann_return_linear_pct']>0).mean()*100:.0f}%"
        )


if __name__ == "__main__":
    main()
