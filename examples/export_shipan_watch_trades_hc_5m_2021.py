# -*- coding: utf-8 -*-
"""导出 shipan 观察模式 5 分钟仓位变化，并与投研 SQhc9001 5m 回测逐笔对比（仅 2021 年）。

用法：
    python examples/export_shipan_watch_trades_hc_5m_2021.py

投研侧：examples/_bt_pen_zone_a1_stop_futures_20210101_20230101_5m（SQhc9001 pairs）
实盘侧：TqSdk 回测 PenZoneA1 hc 主力连续，SHIPAN_FREQ=5分钟（300s 直拉，不 resample）
"""

from __future__ import annotations

import os
import re
import sys
import subprocess
from typing import Dict, List, Optional, Tuple

import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")
os.environ.setdefault("czsc_max_bi_num", "20")

from czsc.connectors.research import get_raw_bars  # noqa: E402
from czsc.traders.dummy import DummyBacktest  # noqa: E402
from czsc.signals.strategies_pen_zone_a1 import (  # noqa: E402
    PenZoneA1Strategy,
    pen_zone_a1_signal_v1,
)

SDT = "20210101"
EDT = "20220101"
FREQ = "5分钟"
ATR_PERIOD = 14
IN_SL = 6
RESEARCH_SYMBOL = "SQhc9001"
SHIPAN_SYMBOL = "KQ.m@SHFE.hc"
POS_NAME = "PenZoneA1V1"
OUT_DIR = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results")
WATCH_CSV = os.path.join(OUT_DIR, "shipan_watch_pos_changes_hc_5m_2021.csv")
COMPARE_CSV = os.path.join(OUT_DIR, "shipan_vs_research_hc_trades_compare_5m_2021.csv")
BT_ROOT_TAG = "_bt_pen_zone_a1_stop_futures_20210101_20230101_5m"
COMPARE_TOL = pd.Timedelta(minutes=5)


class PenZoneA1StrategyHc5m(PenZoneA1Strategy):
    base_freq = FREQ

    @property
    def signals_config(self):
        return [
            {
                "name": pen_zone_a1_signal_v1,
                "freq": self.base_freq,
                "di": 1,
                "log": False,
                "atr_period": ATR_PERIOD,
                "in_sl": IN_SL,
            },
        ]


def kill_duplicate_shipan_processes() -> int:
    """结束其他 shipan_1 / export_shipan 进程，保留当前 PID。"""
    my_pid = os.getpid()
    killed = 0
    if sys.platform != "win32":
        return killed
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return killed
    for line in out.splitlines():
        if "shipan_1" not in line and "export_shipan_watch_trades_hc" not in line:
            continue
        m = re.search(r"(\d+)\s*$", line.strip())
        if not m:
            continue
        pid = int(m.group(1))
        if pid == my_pid:
            continue
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False, capture_output=True)
            killed += 1
        except Exception:
            pass
    return killed


def _pairs_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    for c in df.columns:
        if keyword in str(c):
            return c
    return None


def _filter_2021_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "open_dt" not in df.columns:
        return df
    sdt_ts = pd.Timestamp(SDT)
    edt_ts = pd.Timestamp(EDT)
    out = df.copy()
    out["open_dt"] = pd.to_datetime(out["open_dt"], errors="coerce")
    return out[(out["open_dt"] >= sdt_ts) & (out["open_dt"] < edt_ts)].reset_index(drop=True)


def pos_changes_to_trades(df: pd.DataFrame) -> pd.DataFrame:
    """将仓位变动序列转为开平仓对（与 DummyBacktest pairs 可比）。"""
    if df.empty:
        return pd.DataFrame(
            columns=["trade_idx", "direction", "open_dt", "close_dt", "open_signal", "close_signal"]
        )

    rows = df.sort_values("dt").reset_index(drop=True)
    trades: List[Dict] = []
    open_dt = None
    open_signal = ""
    direction = ""
    trade_idx = 0

    def _dir_from_pos(pos: int) -> str:
        return "多头" if pos > 0 else "空头"

    def _close(close_dt, close_signal: str):
        nonlocal open_dt, open_signal, direction, trade_idx
        if open_dt is None:
            return
        trades.append(
            {
                "trade_idx": trade_idx,
                "direction": direction,
                "open_dt": open_dt,
                "close_dt": close_dt,
                "open_signal": open_signal,
                "close_signal": close_signal,
            }
        )
        trade_idx += 1
        open_dt = None
        open_signal = ""
        direction = ""

    for _, r in rows.iterrows():
        pb, pa = int(r["pos_before"]), int(r["pos_after"])
        dt = pd.Timestamp(r["dt"])
        sig = str(r.get("signal") or "")
        if pb == 0 and pa != 0:
            open_dt = dt
            open_signal = sig
            direction = _dir_from_pos(pa)
        elif pb != 0 and pa == 0:
            _close(dt, sig)
        elif pb != 0 and pa != 0 and pb != pa:
            _close(dt, sig)
            open_dt = dt
            open_signal = sig
            direction = _dir_from_pos(pa)

    return pd.DataFrame(trades)


def run_research_backtest() -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = os.path.join(_EXAMPLES_DIR, BT_ROOT_TAG)
    signals_path = os.path.join(root, "signals")
    results_path = os.path.join(root, "results")
    poss_symbol = os.path.join(results_path, "poss", RESEARCH_SYMBOL)
    pairs_path = os.path.join(poss_symbol, f"{POS_NAME}.pairs")
    holds_path = os.path.join(poss_symbol, f"{POS_NAME}.holds")

    if not os.path.exists(pairs_path):
        bars = get_raw_bars(RESEARCH_SYMBOL, freq=FREQ, sdt=SDT, edt=EDT, fq="后复权")
        if not bars:
            raise RuntimeError(f"投研无 K 线: {RESEARCH_SYMBOL} {FREQ} {SDT}-{EDT}")
        dummy = DummyBacktest(
            strategy=PenZoneA1StrategyHc5m,
            read_bars=get_raw_bars,
            signals_module_name="czsc.signals",
            sdt=SDT,
            edt=EDT,
            signals_path=signals_path,
            results_path=results_path,
        )
        print(f"投研回测: {RESEARCH_SYMBOL} | {SDT}~{EDT} | {FREQ}")
        dummy.execute([RESEARCH_SYMBOL], n_jobs=1)

    dfp = pd.read_parquet(pairs_path)
    dfh = pd.read_parquet(holds_path) if os.path.exists(holds_path) else pd.DataFrame()
    return dfp, dfh


def research_pairs_to_trades(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp.empty:
        return pd.DataFrame(
            columns=["trade_idx", "direction", "open_dt", "close_dt", "open_px", "close_px", "event_seq"]
        )
    open_col = _pairs_col(dfp, "开仓时间")
    close_col = _pairs_col(dfp, "平仓时间")
    dir_col = _pairs_col(dfp, "交易方向")
    open_px_col = _pairs_col(dfp, "开仓价格")
    close_px_col = _pairs_col(dfp, "平仓价格")
    evt_col = _pairs_col(dfp, "事件序列")
    out = pd.DataFrame(
        {
            "trade_idx": range(len(dfp)),
            "direction": dfp[dir_col].astype(str) if dir_col else "",
            "open_dt": pd.to_datetime(dfp[open_col], errors="coerce") if open_col else pd.NaT,
            "close_dt": pd.to_datetime(dfp[close_col], errors="coerce") if close_col else pd.NaT,
            "open_px": dfp[open_px_col] if open_px_col else None,
            "close_px": dfp[close_px_col] if close_px_col else None,
            "event_seq": dfp[evt_col].astype(str) if evt_col else "",
        }
    )
    out = out.sort_values("open_dt").reset_index(drop=True)
    return _filter_2021_trades(out)


def compare_trades(shipan_trades: pd.DataFrame, research_trades: pd.DataFrame) -> pd.DataFrame:
    tol = COMPARE_TOL
    used_shipan = set()
    rows = []
    n = max(len(shipan_trades), len(research_trades))
    s_list = shipan_trades.to_dict("records")
    r_list = research_trades.to_dict("records")

    for i in range(n):
        r = r_list[i] if i < len(r_list) else {}
        best_j = None
        best_delta = None
        if r:
            r_open = pd.Timestamp(r.get("open_dt"))
            r_dir = str(r.get("direction", ""))
            for j, s in enumerate(s_list):
                if j in used_shipan:
                    continue
                s_open = pd.Timestamp(s.get("open_dt"))
                delta = abs(s_open - r_open)
                if delta <= tol and str(s.get("direction", "")) == r_dir:
                    if best_delta is None or delta < best_delta:
                        best_j = j
                        best_delta = delta
        s = s_list[best_j] if best_j is not None else (s_list[i] if i < len(s_list) and i not in used_shipan else {})
        if best_j is not None:
            used_shipan.add(best_j)

        if r and s:
            s_open = pd.Timestamp(s.get("open_dt"))
            r_open = pd.Timestamp(r.get("open_dt"))
            s_close = pd.Timestamp(s.get("close_dt"))
            r_close = pd.Timestamp(r.get("close_dt"))
            open_match = abs(s_open - r_open) <= tol
            close_match = abs(s_close - r_close) <= tol
            dir_match = str(s.get("direction", "")) == str(r.get("direction", ""))
            if open_match and close_match and dir_match:
                status = "match"
            elif open_match and dir_match:
                status = "close_dt_mismatch"
            elif dir_match:
                status = "open_dt_mismatch"
            else:
                status = "mismatch"
        elif r and not s:
            status = "research_only"
        elif s and not r:
            status = "shipan_only"
        else:
            status = "empty"

        rows.append(
            {
                "pair_idx": i + 1,
                "match_status": status,
                "shipan_direction": s.get("direction", ""),
                "research_direction": r.get("direction", ""),
                "shipan_open_dt": s.get("open_dt", ""),
                "research_open_dt": r.get("open_dt", ""),
                "open_dt_delta_min": (
                    round((pd.Timestamp(s.get("open_dt")) - pd.Timestamp(r.get("open_dt"))).total_seconds() / 60, 1)
                    if s.get("open_dt") and r.get("open_dt")
                    else ""
                ),
                "shipan_close_dt": s.get("close_dt", ""),
                "research_close_dt": r.get("close_dt", ""),
                "close_dt_delta_min": (
                    round((pd.Timestamp(s.get("close_dt")) - pd.Timestamp(r.get("close_dt"))).total_seconds() / 60, 1)
                    if s.get("close_dt") and r.get("close_dt")
                    else ""
                ),
                "shipan_open_signal": s.get("open_signal", ""),
                "research_event_seq": r.get("event_seq", ""),
                "research_open_px": r.get("open_px", ""),
                "research_close_px": r.get("close_px", ""),
            }
        )

    for j, s in enumerate(s_list):
        if j in used_shipan:
            continue
        rows.append(
            {
                "pair_idx": len(rows) + 1,
                "match_status": "shipan_only",
                "shipan_direction": s.get("direction", ""),
                "research_direction": "",
                "shipan_open_dt": s.get("open_dt", ""),
                "research_open_dt": "",
                "open_dt_delta_min": "",
                "shipan_close_dt": s.get("close_dt", ""),
                "research_close_dt": "",
                "close_dt_delta_min": "",
                "shipan_open_signal": s.get("open_signal", ""),
                "research_event_seq": "",
                "research_open_px": "",
                "research_close_px": "",
            }
        )
    return pd.DataFrame(rows)


def run_shipan_watch() -> pd.DataFrame:
    os.environ["SHIPAN_WATCH_ONLY"] = "1"
    os.environ["SHIPAN_VERBOSE"] = "0"
    os.environ["SHIPAN_STRATEGY"] = "a1"
    os.environ["SHIPAN_FREQ"] = FREQ
    os.environ.setdefault("SHIPAN_ADJ", "F")

    from examples.shipan_1 import run_multi_main_symbol  # noqa: WPS433

    print(f"shipan 观察模式: {SHIPAN_SYMBOL} | {SDT}~{EDT} | {FREQ}")
    result = run_multi_main_symbol(
        sdt=SDT,
        edt=EDT,
        freq=FREQ,
        watch_only=True,
        verbose=False,
    )
    df = pd.DataFrame(result.get("pos_changes", []))
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"])
    print(
        f"shipan 汇总: pos_changes={result['stats']['pos_changes']} "
        f"final_pos={result['stats']['final_pos']}"
    )
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    killed = kill_duplicate_shipan_processes()
    if killed:
        print(f"已结束 {killed} 个重复 shipan 进程")

    watch_df = run_shipan_watch()
    watch_df.to_csv(WATCH_CSV, index=False, encoding="utf-8-sig")
    print(f"已导出: {WATCH_CSV} ({len(watch_df)} 行仓位变动)")

    dfp, _ = run_research_backtest()
    research_trades = research_pairs_to_trades(dfp)
    shipan_trades = pos_changes_to_trades(watch_df)
    compare_df = compare_trades(shipan_trades, research_trades)
    compare_df.to_csv(COMPARE_CSV, index=False, encoding="utf-8-sig")
    print(f"已导出: {COMPARE_CSV}")

    n_shipan = len(shipan_trades)
    n_research = len(research_trades)
    n_match = int(compare_df["match_status"].eq("match").sum())
    match_rate = (n_match / max(n_shipan, n_research) * 100) if max(n_shipan, n_research) else 0.0
    first3_match = (
        compare_df.head(3)["match_status"].eq("match").all() if len(compare_df) >= 3 else False
    )

    print("\n========== 对比摘要 (5m / 2021) ==========")
    print(f"shipan 仓位变动: {len(watch_df)}")
    print(f"shipan 成交笔数: {n_shipan}")
    print(f"投研成交笔数:   {n_research}")
    print(f"完全匹配笔数:   {n_match}")
    print(f"匹配率:         {match_rate:.1f}%")
    print(f"前 3 笔均匹配:  {first3_match}")
    if not compare_df.empty:
        print(compare_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
