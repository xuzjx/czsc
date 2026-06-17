# -*- coding: utf-8 -*-
"""515880.SH 通信ETF 2025 日线：PenZoneA1EtfV1 有/无 ATR 止损对比（T+1 只做多，年末强平）"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("czsc_max_bi_num", "20")

import czsc  # noqa: E402
from czsc.traders.dummy import DummyBacktest  # noqa: E402
from czsc import Position  # noqa: E402
from czsc.signals.strategies_pen_zone_a1 import (  # noqa: E402
    PenZoneA1EtfStrategy,
    get_events_pen_zone_a1_etf,
    pen_zone_a1_signal_v1,
)
from examples.pairs_year_end import load_pairs_with_year_end_close  # noqa: E402
from examples.return_metrics import total_return_pct  # noqa: E402
from examples.portfolio_nav_from_holds import (  # noqa: E402
    NavConfig,
    annualized_return,
    infer_asset_spec,
    max_drawdown,
    simulate_symbol_nav,
)

ZHIYUE_DIR = os.path.join(_REPO_ROOT, "指月自选etf")
SYMBOL = "515880.SH"
CN_NAME = "通信ETF国泰"
SDT = "20250101"
EDT = "20260101"
FREQ = "日线"
POS_NAME = "PenZoneA1EtfV1"
YEAR_START = pd.Timestamp("2025-01-01")
YEAR_END = pd.Timestamp("2025-12-31 23:59:59")

SCENARIOS = [
    {
        "key": "with_atr",
        "label": "PenZoneA1EtfV1+ATR止损",
        "root_tag": "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily",
        "run_if_missing": False,
    },
    {
        "key": "no_atr",
        "label": "PenZoneA1EtfV1无ATR止损",
        "root_tag": "_bt_pen_zone_a1_no_atr_etf_t1_515880SH_20250101_20260101_daily",
        "run_if_missing": True,
    },
]

OUTPUT_CSV = os.path.join(
    _EXAMPLES_DIR, "_portfolio_nav_results", "515880_daily_strategy_compare_2025.csv"
)


def get_raw_bars_zhiyue(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    raw_bars = kwargs.get("raw_bars", True)
    matches = glob.glob(os.path.join(ZHIYUE_DIR, f"{symbol}.parquet"))
    if not matches:
        return []
    kline = pd.read_parquet(matches[0])
    if "dt" not in kline.columns:
        kline["dt"] = pd.to_datetime(kline["datetime"])
    kline = kline[(kline["dt"] >= pd.to_datetime(sdt)) & (kline["dt"] <= pd.to_datetime(edt))]
    if kline.empty:
        return []
    target = czsc.Freq(freq)
    return czsc.resample_bars(kline, target, raw_bars=raw_bars, base_freq="日线")


class PenZoneA1EtfStrategyDaily(PenZoneA1EtfStrategy):
    base_freq = FREQ


class PenZoneA1EtfStrategyDailyNoAtr(PenZoneA1EtfStrategy):
    """T+1 只做多，关闭 ATR 止损（仅三卖平多）。"""

    base_freq = FREQ

    @property
    def positions(self) -> List[Position]:
        e = get_events_pen_zone_a1_etf(self.base_freq)
        return [
            Position(symbol=self.symbol, name=POS_NAME, opens=[e[0]], exits=[e[1]], T0=False),
        ]

    @property
    def signals_config(self):
        return [
            {
                "name": pen_zone_a1_signal_v1,
                "freq": self.base_freq,
                "di": 1,
                "log": True,
                "atr_period": 14,
                "in_sl": 6,
                "long_only": True,
                "use_atr_stop": False,
            },
        ]


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


def _filter_pairs_year(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty:
        return dfp
    open_col, close_col = _pairs_time_columns(dfp)
    if not open_col or not close_col:
        return dfp.iloc[0:0]
    open_t = pd.to_datetime(dfp[open_col], errors="coerce")
    close_t = pd.to_datetime(dfp[close_col], errors="coerce")
    mask = (open_t >= YEAR_START) & (open_t <= YEAR_END) & (close_t >= YEAR_START) & (close_t <= YEAR_END)
    return dfp.loc[mask].copy()


def _run_no_atr_backtest(bt_root: str) -> None:
    signals_path = os.path.join(bt_root, "signals")
    results_path = os.path.join(bt_root, "results")
    os.makedirs(signals_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    dummy = DummyBacktest(
        strategy=PenZoneA1EtfStrategyDailyNoAtr,
        read_bars=get_raw_bars_zhiyue,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )
    print(f"回测: {SYMBOL} | PenZoneA1EtfV1 无ATR | {SDT}~{EDT} | {FREQ}")
    print(f"输出: {results_path}")
    dummy.execute([SYMBOL], n_jobs=1)


def _scenario_metrics(bt_root: str, label: str) -> Dict:
    pairs_path = os.path.join(bt_root, "results", "poss", SYMBOL, f"{POS_NAME}.pairs")
    holds_path = os.path.join(bt_root, "results", "poss", SYMBOL, f"{POS_NAME}.holds")

    dfp, _ = load_pairs_with_year_end_close(pairs_path, holds_path, YEAR_END, POS_NAME)
    dfp = _filter_pairs_year(dfp)

    bp_col = _pairs_bp_column(dfp) if not dfp.empty else None
    bp = pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).to_numpy() if not dfp.empty else np.array([])
    trades = int(len(bp))
    win_pct = float((bp > 0).mean() * 100.0) if trades else np.nan
    cum_bp = float(bp.sum()) if trades else 0.0
    ret_pct = total_return_pct(cum_bp)

    open_col, close_col = _pairs_time_columns(dfp) if not dfp.empty else (None, None)
    trade_lines: List[str] = []
    if not dfp.empty and open_col and close_col:
        for _, row in dfp.iterrows():
            trade_lines.append(
                f"{pd.Timestamp(row[open_col]).date()}@{row.get('开仓价格', row.iloc[5])}"
                f" -> {pd.Timestamp(row[close_col]).date()}@{row.get('平仓价格', row.iloc[6])}"
                f" ({float(row[bp_col]):+.0f}bp)"
            )

    nav_res = {}
    if os.path.exists(holds_path):
        dfh = pd.read_parquet(holds_path)
        dfh["dt"] = pd.to_datetime(dfh["dt"])
        dfh = dfh[(dfh["dt"] >= YEAR_START) & (dfh["dt"] <= YEAR_END)]
        if not dfh.empty:
            cfg = NavConfig(initial_capital=1_000_000.0, pos_name=POS_NAME)
            spec = infer_asset_spec(SYMBOL)
            nav_df = simulate_symbol_nav(dfh, cfg.initial_capital, cfg, spec)
            if not nav_df.empty:
                nav_start = float(nav_df["nav"].iloc[0])
                nav_end = float(nav_df["nav"].iloc[-1])
                sdt = nav_df["dt"].iloc[0]
                edt = nav_df["dt"].iloc[-1]
                years = (edt - sdt).total_seconds() / (365.25 * 24 * 3600)
                nav_res = {
                    "total_return_pct": (nav_end / nav_start - 1) * 100 if nav_start else np.nan,
                    "ann_return_pct": annualized_return(nav_start, nav_end, years),
                    "max_dd_pct": max_drawdown(nav_df["nav"]),
                }

    return {
        "策略": label,
        "代码": SYMBOL,
        "名称": CN_NAME,
        "频率": FREQ,
        "T+1": "是",
        "ATR止损": "是" if "ATR止损" in label and "无" not in label else "否",
        "交易次数": trades,
        "胜率%": round(win_pct, 2) if np.isfinite(win_pct) else np.nan,
        "累计BP": round(cum_bp, 2),
        "2025年收益%": round(ret_pct, 4),
        "NAV总收益%": round(float(nav_res.get("total_return_pct", np.nan)), 4)
        if nav_res.get("total_return_pct") is not None
        else np.nan,
        "NAV年化%": round(float(nav_res.get("ann_return_pct", np.nan)), 4)
        if nav_res.get("ann_return_pct") is not None
        else np.nan,
        "最大回撤%": round(float(nav_res.get("max_dd_pct", np.nan)), 4)
        if nav_res.get("max_dd_pct") is not None
        else np.nan,
        "开平仓明细": " | ".join(trade_lines),
        "回测目录": bt_root,
        "pairs文件": pairs_path,
        "holds文件": holds_path,
    }


def main():
    rows = []
    for sc in SCENARIOS:
        bt_root = os.path.join(_EXAMPLES_DIR, sc["root_tag"])
        pairs_path = os.path.join(bt_root, "results", "poss", SYMBOL, f"{POS_NAME}.pairs")
        if sc["run_if_missing"] and not os.path.exists(pairs_path):
            _run_no_atr_backtest(bt_root)
        if not os.path.exists(pairs_path):
            print(f"跳过 {sc['label']}: 无 pairs -> {pairs_path}")
            continue
        rows.append(_scenario_metrics(bt_root, sc["label"]))

    if not rows:
        raise SystemExit("无可用回测结果")

    summary_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n========== 515880.SH 2025 日线策略对比 ==========")
    show_cols = [
        "策略",
        "交易次数",
        "胜率%",
        "2025年收益%",
        "NAV总收益%",
        "NAV年化%",
        "最大回撤%",
        "开平仓明细",
    ]
    print(summary_df[show_cols].to_string(index=False))
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
