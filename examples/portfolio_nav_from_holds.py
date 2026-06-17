# -*- coding: utf-8 -*-
"""从 DummyBacktest 输出的 .holds 文件构建组合净值（NAV）。

支持：
- 初始资金、标的等权分配
- ETF/股票按股数、期货按手数 + 保证金
- 手续费与滑点
- 逐 bar 更新 NAV
- 与错误汇总法、截面等权 BP 法对比

用法：
    python examples/portfolio_nav_from_holds.py
    python examples/portfolio_nav_from_holds.py --root examples/_bt_pen_zone_a1_stop_etf5_20210101_20230101_15m
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_POS_NAME = "PenZoneA1V1"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_FEE_RATE = 0.0003  # 单边费率（比例）
DEFAULT_SLIPPAGE = 0.0001  # 单边滑点（比例）

# 期货品种默认合约乘数与保证金率（holds 中无乘数信息时的假设）
# product_code -> (multiplier, margin_rate)
FUTURES_SPECS: Dict[str, Tuple[float, float]] = {
    "rb": (10, 0.13),
    "hc": (10, 0.13),
    "i": (100, 0.14),
    "j": (100, 0.14),
    "jm": (60, 0.14),
    "y": (10, 0.10),
    "p": (10, 0.10),
    "m": (10, 0.10),
    "a": (10, 0.10),
    "c": (10, 0.10),
    "cs": (10, 0.10),
    "jd": (10, 0.10),
    "l": (5, 0.11),
    "v": (5, 0.11),
    "pp": (5, 0.11),
    "eg": (10, 0.11),
    "eb": (5, 0.11),
    "pg": (20, 0.11),
    "fu": (10, 0.13),
    "bu": (10, 0.13),
    "ru": (10, 0.13),
    "sp": (10, 0.13),
    "ss": (5, 0.13),
    "ni": (1, 0.14),
    "cu": (5, 0.14),
    "al": (5, 0.14),
    "zn": (5, 0.14),
    "pb": (5, 0.14),
    "sn": (1, 0.14),
    "au": (1000, 0.10),
    "ag": (15, 0.12),
    "IF": (300, 0.12),
    "IH": (300, 0.12),
    "IC": (200, 0.12),
    "IM": (200, 0.12),
    "T": (10000, 0.02),
    "TF": (10000, 0.012),
    "TS": (20000, 0.005),
    "AP": (10, 0.12),
    "CF": (5, 0.10),
    "CY": (5, 0.10),
    "SR": (10, 0.10),
    "TA": (5, 0.10),
    "MA": (10, 0.10),
    "OI": (10, 0.10),
    "RM": (10, 0.10),
    "ZC": (100, 0.12),
    "FG": (20, 0.10),
    "SA": (20, 0.12),
    "UR": (20, 0.10),
    "PF": (5, 0.10),
    "PK": (5, 0.10),
    "SF": (5, 0.12),
    "SM": (5, 0.12),
    "CJ": (5, 0.12),
    "LH": (16, 0.12),
    "sc": (1000, 0.12),
    "nr": (10, 0.12),
    "lu": (10, 0.12),
    "bc": (5, 0.14),
}
DEFAULT_FUTURES_SPEC = (10, 0.12)


@dataclass
class AssetSpec:
    asset_type: str  # "stock" | "futures"
    multiplier: float = 1.0
    margin_rate: float = 1.0  # stock: 全额资金


@dataclass
class NavConfig:
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    fee_rate: float = DEFAULT_FEE_RATE
    slippage: float = DEFAULT_SLIPPAGE
    pos_name: str = DEFAULT_POS_NAME
    # ETF/股票默认费率可更低
    stock_fee_rate: Optional[float] = None
    stock_slippage: Optional[float] = None
    futures_fee_rate: Optional[float] = None
    futures_slippage: Optional[float] = None


def _parse_futures_product(symbol: str) -> str:
    """SQrb9001 -> rb, ZZIF9001 -> IF"""
    body = symbol[:-4] if symbol.endswith("9001") else symbol
    return re.sub(r"^[A-Z]{2}", "", body)


def infer_asset_spec(symbol: str) -> AssetSpec:
    if symbol.endswith((".SH", ".SZ", ".BJ")):
        return AssetSpec(asset_type="stock", multiplier=1.0, margin_rate=1.0)
    if symbol.endswith("9001"):
        prod = _parse_futures_product(symbol)
        mult, margin = FUTURES_SPECS.get(prod, DEFAULT_FUTURES_SPEC)
        return AssetSpec(asset_type="futures", multiplier=mult, margin_rate=margin)
    # 默认按股票处理
    return AssetSpec(asset_type="stock", multiplier=1.0, margin_rate=1.0)


def list_holds_files(poss_path: str, pos_name: str) -> List[Tuple[str, str]]:
    files = []
    if not os.path.isdir(poss_path):
        return files
    for symbol in sorted(os.listdir(poss_path)):
        sym_path = os.path.join(poss_path, symbol)
        if not os.path.isdir(sym_path):
            continue
        f = os.path.join(sym_path, f"{pos_name}.holds")
        if os.path.exists(f):
            files.append((symbol, f))
    return files


def load_all_holds(poss_path: str, pos_name: str) -> pd.DataFrame:
    rows = []
    for symbol, fpath in list_holds_files(poss_path, pos_name):
        df = pd.read_parquet(fpath)
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["dt", "pos", "price", "n1b", "symbol"])
    out = pd.concat(rows, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"])
    return out.sort_values(["symbol", "dt"]).reset_index(drop=True)


def _fee_slippage(cfg: NavConfig, asset_type: str) -> Tuple[float, float]:
    if asset_type == "stock":
        fee = cfg.stock_fee_rate if cfg.stock_fee_rate is not None else cfg.fee_rate
        slip = cfg.stock_slippage if cfg.stock_slippage is not None else cfg.slippage
    else:
        fee = cfg.futures_fee_rate if cfg.futures_fee_rate is not None else cfg.fee_rate
        slip = cfg.futures_slippage if cfg.futures_slippage is not None else cfg.slippage
    return fee, slip


def compute_position_size(
    alloc: float, price: float, pos: int, spec: AssetSpec
) -> int:
    """根据分配资金计算股数/手数（pos=0 时返回 0）。"""
    if pos == 0 or price <= 0 or alloc <= 0:
        return 0
    if spec.asset_type == "stock":
        return int(alloc // price)
    margin_per_lot = price * spec.multiplier * spec.margin_rate
    if margin_per_lot <= 0:
        return 0
    return int(alloc // margin_per_lot)


def notional(size: int, price: float, spec: AssetSpec) -> float:
    return size * price * spec.multiplier


def simulate_symbol_nav(
    dfh: pd.DataFrame,
    alloc: float,
    cfg: NavConfig,
    spec: AssetSpec,
) -> pd.DataFrame:
    """单标的逐 bar 子账户净值（权益法，向量化）。"""
    fee_rate, slip = _fee_slippage(cfg, spec.asset_type)
    cost_rate = fee_rate + slip

    df = dfh.sort_values("dt").reset_index(drop=True)
    pos = df["pos"].astype(int).to_numpy()
    price = df["price"].astype(float).to_numpy()
    n1b = df["n1b"].astype(float).to_numpy()
    n = len(df)

    size = np.zeros(n, dtype=int)
    trade_cost = np.zeros(n, dtype=float)
    prev_pos = 0
    curr_size = 0

    for i in range(n):
        p = int(pos[i])
        if p != prev_pos:
            cost = 0.0
            if prev_pos != 0 and curr_size > 0:
                cost += notional(curr_size, price[i], spec) * cost_rate
                curr_size = 0
            if p != 0:
                curr_size = compute_position_size(alloc, price[i], p, spec)
                if curr_size > 0:
                    cost += notional(curr_size, price[i], spec) * cost_rate
                else:
                    pos[i] = 0
                    p = 0
            trade_cost[i] = cost
            prev_pos = p
        size[i] = curr_size

    pnl = np.where(
        (pos != 0) & (size > 0),
        size * price * spec.multiplier * (n1b / 10000.0) * pos,
        0.0,
    )
    nav = alloc + np.cumsum(pnl - trade_cost)

    return pd.DataFrame(
        {
            "dt": df["dt"].values,
            "pos": pos,
            "size": size,
            "price": price,
            "pnl": pnl,
            "trade_cost": trade_cost,
            "nav": nav,
        }
    )


def build_portfolio_nav(
    holds: pd.DataFrame,
    cfg: NavConfig,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """等权组合逐 bar NAV。"""
    symbols = sorted(holds["symbol"].unique())
    n = len(symbols)
    if n == 0:
        return pd.DataFrame(), {}

    alloc = cfg.initial_capital / n
    sym_navs: Dict[str, pd.DataFrame] = {}
    nav_parts = []
    for sym in symbols:
        spec = infer_asset_spec(sym)
        dfh = holds[holds["symbol"] == sym]
        sdf = simulate_symbol_nav(dfh, alloc, cfg, spec)
        sym_navs[sym] = sdf
        part = sdf[["dt", "nav"]].copy()
        part["symbol"] = sym
        nav_parts.append(part)

    long_nav = pd.concat(nav_parts, ignore_index=True)
    nav_wide = long_nav.pivot_table(index="dt", columns="symbol", values="nav", aggfunc="last")
    nav_wide = nav_wide.sort_index().ffill().fillna(alloc)
    nav_wide["nav"] = nav_wide.sum(axis=1)
    nav_wide = nav_wide.reset_index()
    nav_wide["ret"] = nav_wide["nav"].pct_change().fillna(0.0)
    return nav_wide, sym_navs


def cross_section_equal_weight_bp(holds: pd.DataFrame) -> pd.Series:
    """截面等权 BP：每 bar 对所有标的等权，收益 = sum(pos*n1b)/N。"""
    n = holds["symbol"].nunique()
    if n == 0:
        return pd.Series(dtype=float)
    g = holds.groupby("dt", sort=True)
    return g.apply(lambda x: (x["n1b"] * x["pos"]).sum() / n, include_groups=False)


def cross_section_holding_only_bp(holds: pd.DataFrame) -> pd.Series:
    """DummyBacktest 同款：仅对持仓标的等权平均。"""
    g = holds.groupby("dt", sort=True)

    def _bar_ret(x):
        cnt = (x["pos"] != 0).sum()
        if cnt == 0:
            return 0.0
        return (x["n1b"] * x["pos"]).sum() / cnt

    return g.apply(_bar_ret, include_groups=False)


def _pairs_bp_column(df: pd.DataFrame) -> str:
    """盈亏比例列：DummyBacktest pairs 最后一列。"""
    for c in df.columns:
        if "盈亏" in str(c) and "比例" in str(c):
            return c
    return df.columns[-1]


def sum_all_pairs_bp(poss_path: str, pos_name: str) -> float:
    """错误方法：直接累加所有 pairs 盈亏 BP。"""
    total = 0.0
    for symbol, _ in list_holds_files(poss_path, pos_name):
        pairs_path = os.path.join(poss_path, symbol, f"{pos_name}.pairs")
        if not os.path.exists(pairs_path):
            continue
        df = pd.read_parquet(pairs_path)
        if df.empty:
            continue
        bp_col = _pairs_bp_column(df)
        total += pd.to_numeric(df[bp_col], errors="coerce").sum()
    return total


def per_symbol_avg_pairs_bp(poss_path: str, pos_name: str) -> float:
    per_sym = []
    for symbol, _ in list_holds_files(poss_path, pos_name):
        pairs_path = os.path.join(poss_path, symbol, f"{pos_name}.pairs")
        if not os.path.exists(pairs_path):
            continue
        df = pd.read_parquet(pairs_path)
        if df.empty:
            continue
        bp_col = _pairs_bp_column(df)
        per_sym.append(pd.to_numeric(df[bp_col], errors="coerce").sum())
    return float(np.mean(per_sym)) if per_sym else 0.0


def annualized_return(nav_start: float, nav_end: float, years: float) -> float:
    if years <= 0 or nav_start <= 0 or nav_end <= 0:
        return np.nan
    ratio = nav_end / nav_start
    if ratio <= 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ann = ratio ** (1.0 / years) - 1.0
    if not np.isfinite(ann):
        return np.nan
    return float(ann)


def _round_ann_pct(rate: float) -> float:
    """年化收益率转百分比；无效值（nan/inf/complex）返回 nan。"""
    if not np.isfinite(rate):
        return np.nan
    return round(float(rate) * 100, 2)


def max_drawdown(nav: pd.Series) -> float:
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    return float(dd.min()) if len(dd) else 0.0


def analyze_backtest(
    bt_root: str,
    cfg: NavConfig,
    scenario_name: str = "",
) -> dict:
    poss_path = os.path.join(bt_root, "results", "poss")
    holds = load_all_holds(poss_path, cfg.pos_name)
    if holds.empty:
        return {"scenario": scenario_name, "error": "no holds"}

    nav_df, _ = build_portfolio_nav(holds, cfg)
    if nav_df.empty:
        return {"scenario": scenario_name, "error": "empty nav"}

    sdt = nav_df["dt"].iloc[0]
    edt = nav_df["dt"].iloc[-1]
    years = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)
    nav_start = cfg.initial_capital
    nav_end = float(nav_df["nav"].iloc[-1])

    eq_bp = cross_section_equal_weight_bp(holds)
    eq_cum_bp = float(eq_bp.sum())
    eq_ann = annualized_return(1.0, 1.0 + eq_cum_bp / 10000.0, years)

    hold_only_bp = cross_section_holding_only_bp(holds)
    hold_only_cum = float(hold_only_bp.sum())

    wrong_sum_bp = sum_all_pairs_bp(poss_path, cfg.pos_name)
    sym_avg_bp = per_symbol_avg_pairs_bp(poss_path, cfg.pos_name)

    ann = annualized_return(nav_start, nav_end, years)
    sym_ann = annualized_return(1.0, 1.0 + sym_avg_bp / 10000.0, years)
    wrong_ann = annualized_return(1.0, 1.0 + wrong_sum_bp / 10000.0, years)

    return {
        "scenario": scenario_name or os.path.basename(bt_root),
        "symbols": holds["symbol"].nunique(),
        "bars": len(nav_df),
        "start": sdt,
        "end": edt,
        "years": round(years, 4),
        "initial_capital": cfg.initial_capital,
        "nav_end": round(nav_end, 2),
        "total_return_pct": round((nav_end / nav_start - 1) * 100, 2) if nav_start > 0 else np.nan,
        "ann_return_pct": _round_ann_pct(ann),
        "max_dd_pct": round(max_drawdown(nav_df["nav"]) * 100, 2),
        "eq_weight_cum_bp": round(eq_cum_bp, 1),
        "eq_weight_ann_pct": _round_ann_pct(eq_ann),
        "hold_only_cum_bp": round(hold_only_cum, 1),
        "wrong_sum_bp": round(wrong_sum_bp, 1),
        "sym_avg_bp": round(sym_avg_bp, 1),
        "sym_avg_ann_pct": _round_ann_pct(sym_ann),
        "wrong_sum_ann_pct": _round_ann_pct(wrong_ann),
        "nav_df": nav_df,
    }


def discover_pen_zone_a1_stop_roots(
    examples_dir: str = _EXAMPLES_DIR,
    etf_only: bool = False,
) -> List[str]:
    roots = []
    for name in sorted(os.listdir(examples_dir)):
        if not name.startswith("_bt_pen_zone_a1_stop"):
            continue
        if etf_only and "etf" not in name.lower():
            continue
        path = os.path.join(examples_dir, name)
        poss = os.path.join(path, "results", "poss")
        if os.path.isdir(poss):
            roots.append(path)
    return roots


def run_comparison(
    roots: List[str],
    cfg: NavConfig,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for root in roots:
        name = os.path.basename(root)
        res = analyze_backtest(root, cfg, scenario_name=name)
        if "error" in res:
            print(f"[skip] {name}: {res['error']}")
            continue

        nav_df = res.pop("nav_df")
        if output_dir:
            nav_df.to_parquet(os.path.join(output_dir, f"{name}_nav.parquet"), index=False)
            nav_df[["dt", "nav"]].to_csv(
                os.path.join(output_dir, f"{name}_nav.csv"), index=False
            )

        rows.append({k: v for k, v in res.items() if k != "nav_df"})

    df = pd.DataFrame(rows)
    if output_dir and not df.empty:
        show_cols = [
            "scenario", "symbols", "years", "nav_end", "total_return_pct", "ann_return_pct",
            "max_dd_pct", "eq_weight_ann_pct", "sym_avg_ann_pct", "wrong_sum_ann_pct",
            "eq_weight_cum_bp", "hold_only_cum_bp", "wrong_sum_bp",
        ]
        df[show_cols].to_excel(os.path.join(output_dir, "portfolio_nav_comparison.xlsx"), index=False)
        df[show_cols].to_csv(os.path.join(output_dir, "portfolio_nav_comparison.csv"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="组合 NAV 回测（基于 holds 文件）")
    parser.add_argument("--root", action="append", help="单个回测目录，可重复指定")
    parser.add_argument("--pos-name", default=DEFAULT_POS_NAME)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--stock-fee", type=float, default=None)
    parser.add_argument("--stock-slippage", type=float, default=None)
    parser.add_argument("--futures-fee", type=float, default=0.00005)
    parser.add_argument("--futures-slippage", type=float, default=0.00002)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results"),
    )
    parser.add_argument(
        "--etf-only",
        action="store_true",
        help="仅运行名称含 etf 的场景（跳过期货 1m 等长耗时目录）",
    )
    args = parser.parse_args()

    cfg = NavConfig(
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        pos_name=args.pos_name,
        stock_fee_rate=args.stock_fee,
        stock_slippage=args.stock_slippage,
        futures_fee_rate=args.futures_fee,
        futures_slippage=args.futures_slippage,
    )

    roots = args.root if args.root else discover_pen_zone_a1_stop_roots(etf_only=args.etf_only)
    if not roots:
        print("未找到 _bt_pen_zone_a1_stop_* 回测目录")
        return

    print(f"初始资金: {cfg.initial_capital:,.0f} | 等权分配 | fee={cfg.fee_rate} slip={cfg.slippage}")
    print(f"期货 fee={cfg.futures_fee_rate} slip={cfg.futures_slippage}")
    print(f"共 {len(roots)} 个场景\n")

    df = run_comparison(roots, cfg, output_dir=args.output_dir)

    if df.empty:
        print("无有效结果")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    show = df[
        [
            "scenario", "symbols", "years",
            "ann_return_pct", "total_return_pct", "max_dd_pct", "nav_end",
            "eq_weight_ann_pct", "sym_avg_ann_pct", "wrong_sum_ann_pct",
        ]
    ]
    print(show.to_string(index=False))
    print(f"\n结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
