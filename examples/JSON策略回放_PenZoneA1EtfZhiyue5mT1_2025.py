# -*- coding: utf-8 -*-
"""PenZoneA1EtfStrategy 指月自选 ETF 5分钟回测 2025（T+1 只做多，ATR 止损）

说明：
- 数据来源：优先使用 `czsc_research_cache` 中的分钟 parquet（通常为 1m/5m 混合），直接读取 5分钟。
- 若缺少 2025 年 5分钟数据，则跳过该标的，并在报告中记录可用范围与数量。

输出目录：
examples/_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_5m
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

import czsc  # noqa: E402
from czsc.traders.dummy import DummyBacktest  # noqa: E402
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1EtfStrategy  # noqa: E402
from czsc.connectors.research import get_raw_bars as get_raw_bars_research  # noqa: E402
from examples.return_metrics import total_return_pct  # noqa: E402
from examples.portfolio_nav_from_holds import NavConfig, analyze_backtest  # noqa: E402
from examples.export_etf_trades_detail import export_trades_detail  # noqa: E402
from examples.pairs_year_end import load_pairs_with_year_end_close  # noqa: E402

ZHIYUE_DIR = os.path.join(_REPO_ROOT, "指月自选etf")
SDT = "20250101"
EDT = "20260101"
FREQ = "5分钟"
N_JOBS = 2
POS_NAME = "PenZoneA1EtfV1"
ROOT_TAG = f"_bt_pen_zone_a1_stop_etf_zhiyue_t1_{SDT}_{EDT}_5m"
YEAR_START = pd.Timestamp("2025-01-01")
YEAR_END = pd.Timestamp("2025-12-31 23:59:59")


def load_zhiyue_symbols() -> Tuple[List[str], Dict[str, str]]:
    codes_path = os.path.join(ZHIYUE_DIR, "codes.tsv")
    df = pd.read_csv(codes_path, sep="\t", dtype=str)
    symbols = df["symbol"].dropna().tolist()
    name_map = dict(zip(df["symbol"], df["name"]))
    return symbols, name_map


def _research_has_minute(symbol: str) -> bool:
    cache = os.environ.get("czsc_research_cache", r"D:\CZSC投研数据")
    files = glob.glob(os.path.join(cache, "*", f"{symbol}.parquet"))
    if not files:
        return False
    dt_ser = pd.to_datetime(pd.read_parquet(files[0], columns=["dt"])["dt"], errors="coerce").dropna()
    if dt_ser.empty or len(dt_ser) < 100:
        return False
    dt_ser = dt_ser.sort_values()
    diffs = dt_ser.diff().dropna()
    if diffs.empty:
        return False
    med_seconds = diffs.median().total_seconds()
    return med_seconds <= 6 * 3600


def probe_symbol_5m(symbol: str) -> Tuple[str, pd.Timestamp | None, pd.Timestamp | None, int, int]:
    """返回 (source, dt_min, dt_max, n_bars_2025, n_bars_all)"""
    if not _research_has_minute(symbol):
        return "no_minute_in_research_cache", None, None, 0, 0
    try:
        bars = get_raw_bars_research(symbol, FREQ, SDT, EDT, fq="前复权")
        if not bars:
            return "research_cache", None, None, 0, 0
        dts = pd.to_datetime([b.dt for b in bars])
        # pd.to_datetime(list[datetime]) -> DatetimeIndex; use .year (not .dt.year)
        n2025 = int((dts.year == 2025).sum())
        return "research_cache", dts.min(), dts.max(), n2025, int(len(bars))
    except Exception:
        return "research_cache_invalid", None, None, 0, 0


def get_raw_bars_zhiyue_5m(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    """5m 数据读取：仅 research_cache；缺失则返回空。"""
    raw_bars = kwargs.get("raw_bars", True)
    target = czsc.Freq(freq)
    if not _research_has_minute(symbol):
        return []
    return get_raw_bars_research(symbol, target, sdt, edt, fq=fq, raw_bars=raw_bars)


def get_raw_bars_zhiyue_5m_entry(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    """DummyBacktest 多进程需要可 pickle 的顶层函数入口。"""
    return get_raw_bars_zhiyue_5m(symbol, freq, sdt, edt, fq=fq, **kwargs)


def _pairs_bp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "盈亏" in str(c) and "比例" in str(c):
            return c
    return df.columns[-1]


def _pairs_time_columns(df: pd.DataFrame):
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


def per_symbol_summary(bt_root: str, name_map: Dict[str, str], all_symbols: List[str]) -> pd.DataFrame:
    poss = os.path.join(bt_root, "results", "poss")
    rows = []
    for symbol in all_symbols:
        pairs_path = os.path.join(poss, symbol, f"{POS_NAME}.pairs")
        holds_path = os.path.join(poss, symbol, f"{POS_NAME}.holds")
        if not os.path.exists(pairs_path):
            rows.append(
                {
                    "场景": "etf_zhiyue_t1_2025_5m",
                    "代码": symbol,
                    "名称": name_map.get(symbol, ""),
                    "交易次数": 0,
                    "胜率%": np.nan,
                    "累计BP": 0.0,
                    "均笔BP": np.nan,
                    "2025年收益%": 0.0,
                }
            )
            continue
        dfp, _ = load_pairs_with_year_end_close(pairs_path, holds_path, YEAR_END, POS_NAME)
        dfp = _filter_pairs_year(dfp)
        bp_col = _pairs_bp_column(dfp) if not dfp.empty else None
        bp = pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).to_numpy() if not dfp.empty else np.array([])
        cum_bp = float(bp.sum())
        trades = int(len(bp))
        win_pct = float((bp > 0).mean() * 100.0) if trades else np.nan
        rows.append(
            {
                "场景": "etf_zhiyue_t1_2025_5m",
                "代码": symbol,
                "名称": name_map.get(symbol, ""),
                "交易次数": trades,
                "胜率%": round(win_pct, 2) if np.isfinite(win_pct) else np.nan,
                "累计BP": round(cum_bp, 2),
                "均笔BP": round(float(bp.mean()), 2) if trades else np.nan,
                "2025年收益%": round(total_return_pct(cum_bp), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("2025年收益%", ascending=False, na_position="last")


def export_trades_with_names(bt_root: str, output_csv: str, name_map: Dict[str, str]) -> pd.DataFrame:
    df = export_trades_detail(
        bt_root=bt_root,
        pos_name=POS_NAME,
        scenario="etf_zhiyue_t1_2025_5m",
        year_filter=True,
    )
    if df.empty:
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return df
    df["cn_name"] = df["symbol"].map(name_map).fillna(df["cn_name"])
    rename = {
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
    df.rename(columns=rename).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def main():
    parser = argparse.ArgumentParser(description="指月ETF 5m 回测（2025，T+1 long-only）")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    symbols, name_map = load_zhiyue_symbols()
    bt_root = os.path.join(_EXAMPLES_DIR, ROOT_TAG)
    signals_path = os.path.join(bt_root, "signals")
    results_path = os.path.join(bt_root, "results")
    summary_dir = os.path.join(bt_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    print(f"回测: PenZoneA1EtfStrategy | {SDT} ~ {EDT} | freq={FREQ} | T+1 long-only")
    print(f"标的数: {len(symbols)} | 输出: {results_path}")

    # --- 数据可用性探测（用于跳过缺失标的 + 输出报告） ---
    probe_rows = []
    run_symbols = []
    for i, sym in enumerate(symbols, start=1):
        source, dt_min, dt_max, n2025, n_all = probe_symbol_5m(sym)
        full_2025 = int(dt_min is not None and dt_min <= YEAR_START and dt_max is not None and dt_max >= YEAR_END and n2025 > 0)
        probe_rows.append(
            {
                "代码": sym,
                "名称": name_map.get(sym, ""),
                "数据源": source,
                "2025可用K线数": int(n2025),
                "全区间K线数": int(n_all),
                "数据起始": "" if dt_min is None else str(dt_min),
                "数据结束": "" if dt_max is None else str(dt_max),
                "覆盖完整2025": full_2025,
            }
        )
        if n2025 > 0 and source == "research_cache":
            run_symbols.append(sym)
        if i % 10 == 0:
            print(f"[probe] {i}/{len(symbols)} done ...")

    probe_df = pd.DataFrame(probe_rows).sort_values(["覆盖完整2025", "2025可用K线数"], ascending=[False, False])
    probe_csv = os.path.join(summary_dir, "data_availability_5m_2025.csv")
    probe_df.to_csv(probe_csv, index=False, encoding="utf-8-sig")

    dummy = DummyBacktest(
        strategy=PenZoneA1EtfStrategy,
        read_bars=get_raw_bars_zhiyue_5m_entry,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )
    print(f"执行回测标的数: {len(run_symbols)} / {len(symbols)} | 仅 2025 有 5m 数据的标的")
    if run_symbols:
        dummy.execute(run_symbols, n_jobs=int(args.n_jobs))
    else:
        print("[skip] 无可用 5m 数据标的，跳过 DummyBacktest.execute")

    # --- 汇总 ---
    sym_df = per_symbol_summary(bt_root, name_map, symbols)
    sym_csv = os.path.join(summary_dir, "etf_zhiyue_t1_per_symbol_2025_5m.csv")
    sym_df.to_csv(sym_csv, index=False, encoding="utf-8-sig")

    trades_csv = os.path.join(summary_dir, "etf_zhiyue_t1_trades_detail_2025_5m.csv")
    trades_df = export_trades_with_names(bt_root, trades_csv, name_map)

    cfg = NavConfig(initial_capital=1_000_000.0, pos_name=POS_NAME)
    nav_res = analyze_backtest(bt_root, cfg, scenario_name="etf_zhiyue_t1_2025_5m")
    nav_res.pop("nav_df", None)

    total_trades = int(pd.to_numeric(sym_df["交易次数"], errors="coerce").fillna(0).sum()) if not sym_df.empty else 0
    sym_avg_ret = float(sym_df["2025年收益%"].mean()) if not sym_df.empty else np.nan
    metrics = {
        "总交易笔数": total_trades,
        "等权品种均2025年收益%": round(sym_avg_ret, 4) if np.isfinite(sym_avg_ret) else np.nan,
        "NAV年化%": nav_res.get("ann_return_pct", np.nan),
        "NAV总收益%": nav_res.get("total_return_pct", np.nan),
        "NAV最大回撤%": nav_res.get("max_dd_pct", np.nan),
        "NAV期末": nav_res.get("nav_end", np.nan),
        "标的数": len(symbols),
        "参与回测标的数": len(run_symbols),
        "覆盖完整2025标的数": int(probe_df["覆盖完整2025"].sum()) if not probe_df.empty else 0,
    }
    metrics_csv = os.path.join(summary_dir, "etf_zhiyue_t1_metrics_2025_5m.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    print("\n========== 5m 回测汇总 (2025) ==========")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved availability -> {probe_csv}")
    print(f"Saved per-symbol   -> {sym_csv}")
    print(f"Saved trades       -> {trades_csv} ({len(trades_df)} rows)")
    print(f"Saved metrics      -> {metrics_csv}")


if __name__ == "__main__":
    main()

