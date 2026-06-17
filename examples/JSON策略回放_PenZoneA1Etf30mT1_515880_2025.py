# -*- coding: utf-8 -*-
"""PenZoneA1EtfStrategy 单标的 515880.SH 30分钟回测 2025（T+1 只做多，ATR 止损）

数据源优先级：
  1. czsc_research_cache 分钟 parquet（515880 当前仅日线，resample 会失败）
  2. akshare Sina 30m（约 1970 根滚动窗口，2025 仅覆盖 2025-06-11 ~ 2025-12-31）
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Optional, Tuple

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
from examples.return_metrics import total_return_pct  # noqa: E402
from examples.export_etf_trades_detail import export_trades_detail  # noqa: E402

SYMBOL = "515880.SH"
SDT = "20250101"
EDT = "20260101"
FREQ = "30分钟"
POS_NAME = "PenZoneA1EtfV1"
ROOT_TAG = f"_bt_pen_zone_a1_stop_etf_t1_515880SH_{SDT}_{EDT}_30m"
YEAR_START = pd.Timestamp("2025-01-01")
YEAR_END = pd.Timestamp("2025-12-31 23:59:59")


def _research_has_minute(symbol: str) -> bool:
    cache = os.environ.get("czsc_research_cache", r"D:\CZSC投研数据")
    files = glob.glob(os.path.join(cache, "*", f"{symbol}.parquet"))
    if not files:
        return False
    dt_ser = pd.to_datetime(pd.read_parquet(files[0], columns=["dt"])["dt"])
    return dt_ser.dt.time.nunique() > 2


def _fetch_30m_akshare(symbol: str, sdt: str, edt: str) -> pd.DataFrame:
    import akshare as ak

    code, exch = symbol.split(".")
    sina = ("sh" if exch.upper() == "SH" else "sz") + code
    raw = ak.stock_zh_a_minute(symbol=sina, period="30", adjust="qfq")
    df = raw.rename(columns={"day": "dt", "volume": "vol"})
    df["dt"] = pd.to_datetime(df["dt"])
    df["symbol"] = symbol
    for col in ("open", "close", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0).astype("int64")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype("float64")
    sdt_ts, edt_ts = pd.to_datetime(sdt), pd.to_datetime(edt)
    return df[(df["dt"] >= sdt_ts) & (df["dt"] <= edt_ts)].copy()


def get_raw_bars_515880_30m(symbol, freq, sdt, edt, fq="前复权", **kwargs):
    raw_bars = kwargs.get("raw_bars", True)
    target = czsc.Freq(freq)

    if _research_has_minute(symbol):
        from czsc.connectors.research import get_raw_bars

        return get_raw_bars(symbol, freq, sdt, edt, fq=fq, **kwargs)

    kline = _fetch_30m_akshare(symbol, sdt, edt)
    if kline.empty:
        return []
    kline = kline[["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]].copy()
    return czsc.resample_bars(kline, target, raw_bars=raw_bars, base_freq="30分钟")


def probe_data_source() -> Tuple[str, pd.DataFrame]:
    if _research_has_minute(SYMBOL):
        from czsc.connectors.research import get_raw_bars

        bars = get_raw_bars(SYMBOL, FREQ, SDT, EDT, fq="前复权")
        if bars:
            df = pd.DataFrame({"dt": [b.dt for b in bars]})
            return "research_cache", df
    df = _fetch_30m_akshare(SYMBOL, SDT, EDT)
    return "akshare_sina_30m", df[["dt"]].copy() if not df.empty else df


def _pairs_bp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "盈亏" in str(c) and "比例" in str(c):
            return c
    return df.columns[-1]


def _pairs_time_columns(df: pd.DataFrame):
    open_col, close_col, open_px, close_px = None, None, None, None
    for c in df.columns:
        cs = str(c)
        if "开仓" in cs and "时间" in cs:
            open_col = c
        if "平仓" in cs and "时间" in cs:
            close_col = c
        if "开仓" in cs and "价格" in cs:
            open_px = c
        if "平仓" in cs and "价格" in cs:
            close_px = c
    return open_col, close_col, open_px, close_px


def _filter_pairs_year(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty:
        return dfp
    open_col, close_col, _, _ = _pairs_time_columns(dfp)
    if not open_col or not close_col:
        return dfp.iloc[0:0]
    open_t = pd.to_datetime(dfp[open_col], errors="coerce")
    close_t = pd.to_datetime(dfp[close_col], errors="coerce")
    mask = (open_t >= YEAR_START) & (open_t <= YEAR_END) & (close_t >= YEAR_START) & (close_t <= YEAR_END)
    return dfp.loc[mask].copy()


def main():
    source, probe_df = probe_data_source()
    bt_root = os.path.join(_EXAMPLES_DIR, ROOT_TAG)
    signals_path = os.path.join(bt_root, "signals")
    results_path = os.path.join(bt_root, "results")
    summary_dir = os.path.join(bt_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    print(f"回测: {SYMBOL} | PenZoneA1EtfStrategy | {SDT} ~ {EDT} | freq={FREQ} | T+1 long-only")
    print(f"数据源: {source}")
    if probe_df.empty:
        print("\n*** 30分钟数据不可用：research cache 仅日线，akshare 也未返回数据 ***")
        print("建议：配置 CZSC_TOKEN / TUSHARE_TOKEN 拉取分钟 parquet，或使用日线回测作参考。")
        return 1

    dt_min, dt_max = probe_df["dt"].min(), probe_df["dt"].max()
    print(f"2025 可用 30m K 线: {len(probe_df)} 根 | {dt_min} ~ {dt_max}")
    if dt_min > YEAR_START:
        print(f"注意: 缺少 {YEAR_START.date()} ~ {(dt_min - pd.Timedelta(days=1)).date()} 的 30m 数据（akshare Sina 滚动窗口限制）")

    dummy = DummyBacktest(
        strategy=PenZoneA1EtfStrategy,
        read_bars=get_raw_bars_515880_30m,
        signals_module_name="czsc.signals",
        sdt=SDT,
        edt=EDT,
        signals_path=signals_path,
        results_path=results_path,
    )
    print(f"输出: {results_path}\n")
    dummy.execute([SYMBOL], n_jobs=1)

    pairs_path = os.path.join(results_path, "poss", SYMBOL, f"{POS_NAME}.pairs")
    dfp = _filter_pairs_year(pd.read_parquet(pairs_path)) if os.path.exists(pairs_path) else pd.DataFrame()
    bp_col = _pairs_bp_column(dfp) if not dfp.empty else None
    bp = pd.to_numeric(dfp[bp_col], errors="coerce").fillna(0.0).to_numpy() if not dfp.empty else np.array([])
    cum_bp = float(bp.sum())
    trades = int(len(bp))
    win_pct = float((bp > 0).mean() * 100.0) if trades else np.nan
    ret_pct = round(total_return_pct(cum_bp), 4)

    open_col, close_col, open_px, close_px = _pairs_time_columns(dfp) if not dfp.empty else (None,) * 4
    trades_csv = os.path.join(summary_dir, "515880_30m_trades_detail_2025.csv")
    trades_df = export_trades_detail(
        bt_root=bt_root,
        pos_name=POS_NAME,
        scenario="515880_30m_t1_2025",
        year_filter=True,
    )
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")

    metrics = {
        "标的": SYMBOL,
        "策略": POS_NAME,
        "周期": FREQ,
        "数据源": source,
        "数据起始": str(dt_min),
        "数据结束": str(dt_max),
        "2025可用K线数": len(probe_df),
        "2025交易笔数": trades,
        "胜率%": round(win_pct, 2) if np.isfinite(win_pct) else np.nan,
        "累计BP": round(cum_bp, 2),
        "2025年收益%": ret_pct,
    }
    metrics_csv = os.path.join(summary_dir, "515880_30m_metrics_2025.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    print("\n========== 515880.SH 30m 回测汇总 (2025) ==========")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if trades:
        print("\n--- 逐笔开平仓 ---")
        for i, row in dfp.iterrows():
            print(
                f"  #{i+1} {row[open_col]} @ {row[open_px]} -> {row[close_col]} @ {row[close_px]} | BP={row[bp_col]:.2f}"
            )

    try:
        stats = dummy.one_pos_stats(POS_NAME)
        if stats:
            print("\n--- DummyBacktest stats ---")
            for k, v in stats.items():
                if k != "pos_dump":
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"汇总统计跳过: {e}")

    print(f"\nSaved metrics -> {metrics_csv}")
    print(f"Saved trades  -> {trades_csv}")
    print(f"Saved pairs   -> {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
