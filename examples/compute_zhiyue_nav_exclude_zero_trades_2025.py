# -*- coding: utf-8 -*-
"""
排除 0 交易标的，重算 2025 组合净值与年化。

默认使用回测目录：
examples/_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily

输出：
examples/_portfolio_nav_results/zhiyue_nav_exclude_zero_trades_2025.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

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
)


DEFAULT_BT_ROOT = os.path.join(
    _EXAMPLES_DIR,
    "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily",
)
DEFAULT_POS_NAME = "PenZoneA1EtfV1"


def _load_trade_symbols(summary_csv: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(summary_csv)
    df["交易次数"] = pd.to_numeric(df["交易次数"], errors="coerce").fillna(0).astype(int)
    traded = df.loc[df["交易次数"] > 0, "代码"].astype(str).tolist()
    zero = df.loc[df["交易次数"] <= 0, "代码"].astype(str).tolist()
    return traded, zero


def _load_holds_for_symbols(poss_path: str, pos_name: str, symbols: List[str]) -> pd.DataFrame:
    rows = []
    missing = []
    for sym in symbols:
        fpath = os.path.join(poss_path, sym, f"{pos_name}.holds")
        if not os.path.exists(fpath):
            missing.append(sym)
            continue
        dfh = pd.read_parquet(fpath)
        if "symbol" not in dfh.columns:
            dfh["symbol"] = sym
        rows.append(dfh)

    if missing:
        print(f"[warn] missing holds ({len(missing)}): {missing}")

    if not rows:
        return pd.DataFrame(columns=["dt", "pos", "price", "n1b", "symbol"])

    out = pd.concat(rows, ignore_index=True)
    out["dt"] = pd.to_datetime(out["dt"])
    return out.sort_values(["symbol", "dt"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="排除 0 交易标的，计算 2025 组合年化")
    parser.add_argument("--root", default=DEFAULT_BT_ROOT, help="回测根目录")
    parser.add_argument("--pos-name", default=DEFAULT_POS_NAME, help="持仓文件名（不含后缀）")
    parser.add_argument(
        "--summary",
        default=os.path.join(DEFAULT_BT_ROOT, "summary", "etf_zhiyue_t1_per_symbol_2025.csv"),
        help="per_symbol 汇总 CSV（用于筛选交易次数>0）",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "zhiyue_nav_exclude_zero_trades_2025.csv"),
        help="输出 CSV 路径",
    )
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0003)
    parser.add_argument("--slippage", type=float, default=0.0001)
    args = parser.parse_args()

    traded_syms, zero_syms = _load_trade_symbols(args.summary)
    poss_path = os.path.join(args.root, "results", "poss")

    holds = _load_holds_for_symbols(poss_path, args.pos_name, traded_syms)
    if holds.empty:
        raise SystemExit("no holds loaded after filtering traded symbols")

    cfg = NavConfig(
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        pos_name=args.pos_name,
        futures_fee_rate=0.00005,
        futures_slippage=0.00002,
    )

    nav_df, _ = build_portfolio_nav(holds, cfg)
    nav_df = nav_df.sort_values("dt").reset_index(drop=True)

    nav_2025 = nav_df[(nav_df["dt"] >= "2025-01-01") & (nav_df["dt"] <= "2025-12-31")].copy()
    if nav_2025.empty:
        raise SystemExit("no nav rows in 2025 after filtering")

    sdt = nav_2025["dt"].iloc[0]
    edt = nav_2025["dt"].iloc[-1]
    years = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)

    nav_start = float(nav_2025["nav"].iloc[0])
    nav_end = float(nav_2025["nav"].iloc[-1])
    total_ret = nav_end / nav_start - 1.0 if nav_start > 0 else np.nan
    ann = annualized_return(nav_start, nav_end, years)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    nav_2025.to_csv(args.output, index=False)

    print(
        "\n".join(
            [
                f"traded_symbols={len(traded_syms)} | zero_trade_symbols={len(zero_syms)}",
                f"nav_start={nav_start:.2f} nav_end={nav_end:.2f} total_return_pct={total_ret*100:.2f}%",
                f"years={years:.4f} ann_return_pct={ann*100:.2f}%",
                f"output={args.output}",
            ]
        )
    )


if __name__ == "__main__":
    main()

