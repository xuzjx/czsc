# -*- coding: utf-8 -*-
"""PenZoneA1EtfStrategy 指月自选 ETF 日线回测 2025（T+1 只做多，ATR 止损）

标的：指月自选etf/codes.tsv（36 只）
K 线：指月自选etf/*.parquet（日线，非 czsc_research_cache）
"""

from __future__ import annotations

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

os.environ.setdefault("czsc_max_bi_num", "20")

import czsc  # noqa: E402
from czsc.traders.dummy import DummyBacktest  # noqa: E402
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1EtfStrategy  # noqa: E402
from examples.return_metrics import total_return_pct  # noqa: E402
from examples.portfolio_nav_from_holds import NavConfig, analyze_backtest  # noqa: E402
from examples.export_etf_trades_detail import export_trades_detail  # noqa: E402
from examples.pairs_year_end import load_pairs_with_year_end_close  # noqa: E402

ZHIYUE_DIR = os.path.join(_REPO_ROOT, "指月自选etf")
SDT = "20250101"
EDT = "20260101"
FREQ = "日线"
N_JOBS = 4
POS_NAME = "PenZoneA1EtfV1"
ROOT_TAG = f"_bt_pen_zone_a1_stop_etf_zhiyue_t1_{SDT}_{EDT}_daily"
YEAR_START = pd.Timestamp("2025-01-01")
YEAR_END = pd.Timestamp("2025-12-31 23:59:59")


class PenZoneA1EtfStrategyDaily(PenZoneA1EtfStrategy):
    base_freq = FREQ


def load_zhiyue_symbols() -> Tuple[List[str], Dict[str, str]]:
    codes_path = os.path.join(ZHIYUE_DIR, "codes.tsv")
    df = pd.read_csv(codes_path, sep="\t", dtype=str)
    symbols = df["symbol"].dropna().tolist()
    name_map = dict(zip(df["symbol"], df["name"]))
    return symbols, name_map


def get_raw_bars_zhiyue(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    """从 指月自选etf 目录读取日线 parquet，转为 RawBar 列表。"""
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
                    "场景": "etf_zhiyue_t1_2025",
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
                "场景": "etf_zhiyue_t1_2025",
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
        scenario="etf_zhiyue_t1_2025",
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


def check_incomplete_2025(symbols: List[str]) -> pd.DataFrame:
    """对照 159901.SZ 2025 交易日历，标记缺 bar 的标的。"""
    ref_path = os.path.join(ZHIYUE_DIR, "159901.SZ.parquet")
    cache_ref = os.path.join(os.environ.get("czsc_research_cache", r"D:\CZSC投研数据"), "A股场内基金", "159901.SZ.parquet")
    ref_file = ref_path if os.path.exists(ref_path) else cache_ref
    if os.path.exists(ref_file):
        ref_dates = set(pd.to_datetime(pd.read_parquet(ref_file, columns=["dt"])["dt"]).dt.date)
        ref_dates = {d for d in ref_dates if d.year == 2025}
    else:
        ref_dates = set()

    rows = []
    for sym in symbols:
        p = os.path.join(ZHIYUE_DIR, f"{sym}.parquet")
        if not os.path.exists(p):
            rows.append({"代码": sym, "2025缺bar数": -1, "2025首bar": "", "2025末bar": "", "完整": 0})
            continue
        dt_ser = pd.to_datetime(pd.read_parquet(p, columns=["dt"])["dt"])
        dates_2025 = {d for d in dt_ser.dt.date if d.year == 2025}
        missing = sorted(ref_dates - dates_2025) if ref_dates else []
        min_d = min(dates_2025).isoformat() if dates_2025 else ""
        max_d = max(dates_2025).isoformat() if dates_2025 else ""
        rows.append(
            {
                "代码": sym,
                "2025缺bar数": len(missing),
                "2025首bar": min_d,
                "2025末bar": max_d,
                "完整": int(len(missing) == 0 and len(dates_2025) == len(ref_dates)) if ref_dates else int(len(dates_2025) > 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    symbols, name_map = load_zhiyue_symbols()
    bt_root = os.path.join(_EXAMPLES_DIR, ROOT_TAG)
    signals_path = os.path.join(bt_root, "signals")
    results_path = os.path.join(bt_root, "results")
    summary_dir = os.path.join(bt_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    print(f"回测: PenZoneA1EtfStrategyDaily | {SDT} ~ {EDT} | freq={FREQ} | T+1 long-only")
    print(f"标的数: {len(symbols)} | K线目录: {ZHIYUE_DIR}")
    print(f"输出: {results_path}")

    dummy = DummyBacktest(
        strategy=PenZoneA1EtfStrategyDaily,
        read_bars=get_raw_bars_zhiyue,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )
    dummy.execute(symbols, n_jobs=N_JOBS)

    # --- 汇总 ---
    sym_df = per_symbol_summary(bt_root, name_map, symbols)
    sym_csv = os.path.join(summary_dir, "etf_zhiyue_t1_per_symbol_2025.csv")
    sym_df.to_csv(sym_csv, index=False, encoding="utf-8-sig")

    trades_csv = os.path.join(summary_dir, "etf_zhiyue_t1_trades_detail_2025.csv")
    trades_df = export_trades_with_names(bt_root, trades_csv, name_map)

    incomplete_df = check_incomplete_2025(symbols)
    incomplete_csv = os.path.join(summary_dir, "data_completeness_2025.csv")
    incomplete_df.to_csv(incomplete_csv, index=False, encoding="utf-8-sig")

    cfg = NavConfig(initial_capital=1_000_000.0, pos_name=POS_NAME)
    nav_res = analyze_backtest(bt_root, cfg, scenario_name="etf_zhiyue_t1_2025")
    nav_res.pop("nav_df", None)

    all_bp = []
    for _, r in sym_df.iterrows():
        if r["交易次数"] > 0:
            pairs_path = os.path.join(results_path, "poss", r["代码"], f"{POS_NAME}.pairs")
            holds_path = os.path.join(results_path, "poss", r["代码"], f"{POS_NAME}.holds")
            dfp, _ = load_pairs_with_year_end_close(pairs_path, holds_path, YEAR_END, POS_NAME)
            dfp = _filter_pairs_year(dfp)
            bp_col = _pairs_bp_column(dfp)
            all_bp.extend(pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).tolist())

    total_trades = int(len(all_bp))
    win_pct = float((np.array(all_bp) > 0).mean() * 100) if total_trades else np.nan
    cum_bp = float(np.sum(all_bp)) if all_bp else 0.0

    sym_avg_ret = float(sym_df["2025年收益%"].mean()) if not sym_df.empty else np.nan
    metrics = {
        "总交易笔数": total_trades,
        "胜率%": round(win_pct, 2) if np.isfinite(win_pct) else np.nan,
        "累计BP": round(cum_bp, 2),
        "等权品种均2025年收益%": round(sym_avg_ret, 4) if np.isfinite(sym_avg_ret) else np.nan,
        "NAV年化%": nav_res.get("ann_return_pct", np.nan),
        "NAV总收益%": nav_res.get("total_return_pct", np.nan),
        "NAV期末": nav_res.get("nav_end", np.nan),
        "标的数": len(symbols),
        "有交易标的数": int((sym_df["交易次数"] > 0).sum()),
    }
    metrics_csv = os.path.join(summary_dir, "etf_zhiyue_t1_metrics_2025.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    print("\n========== 回测汇总 ==========")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if nav_res:
        print(f"  最大回撤%: {nav_res.get('max_dd_pct', np.nan)}")

    bad = incomplete_df[incomplete_df["完整"] == 0]
    print(f"\n2025 数据不完整标的: {len(bad)} / {len(symbols)}")
    if not bad.empty:
        print(bad.to_string(index=False))

    print(f"\nSaved per-symbol -> {sym_csv}")
    print(f"Saved trades     -> {trades_csv} ({len(trades_df)} rows)")
    print(f"Saved metrics    -> {metrics_csv}")
    print(f"Saved completeness -> {incomplete_csv}")


if __name__ == "__main__":
    main()
