# -*- coding: utf-8 -*-
"""
指月ETF：30分钟 vs 日线 回测对比（2025）

依赖：
- 日线回测目录（已存在）：
  examples/_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily/summary/
- 30m 回测目录（由脚本 JSON策略回放_PenZoneA1EtfZhiyue30mT1_2025.py 生成）：
  examples/_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_30m/summary/

输出：
examples/_portfolio_nav_results/zhiyue_compare_30m_vs_daily_2025.csv
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DAILY_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_daily")
M30_ROOT = os.path.join(_EXAMPLES_DIR, "_bt_pen_zone_a1_stop_etf_zhiyue_t1_20250101_20260101_30m")


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str)


def _to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    daily_sym_csv = os.path.join(DAILY_ROOT, "summary", "etf_zhiyue_t1_per_symbol_2025.csv")
    m30_sym_csv = os.path.join(M30_ROOT, "summary", "etf_zhiyue_t1_per_symbol_2025_30m.csv")
    m30_avail_csv = os.path.join(M30_ROOT, "summary", "data_availability_30m_2025.csv")

    daily = _read_csv_if_exists(daily_sym_csv)
    m30 = _read_csv_if_exists(m30_sym_csv)
    avail = _read_csv_if_exists(m30_avail_csv)

    if daily is None:
        raise SystemExit(f"missing daily per-symbol csv: {daily_sym_csv}")
    if m30 is None:
        raise SystemExit(f"missing 30m per-symbol csv: {m30_sym_csv} (please run 30m backtest script first)")

    daily = daily.rename(
        columns={
            "交易次数": "日线_交易次数",
            "胜率%": "日线_胜率%",
            "累计BP": "日线_累计BP",
            "均笔BP": "日线_均笔BP",
            "2025年收益%": "日线_2025年收益%",
        }
    )
    m30 = m30.rename(
        columns={
            "交易次数": "30m_交易次数",
            "胜率%": "30m_胜率%",
            "累计BP": "30m_累计BP",
            "均笔BP": "30m_均笔BP",
            "2025年收益%": "30m_2025年收益%",
        }
    )

    keep_cols_daily = ["代码", "名称", "日线_交易次数", "日线_胜率%", "日线_累计BP", "日线_均笔BP", "日线_2025年收益%"]
    keep_cols_m30 = ["代码", "30m_交易次数", "30m_胜率%", "30m_累计BP", "30m_均笔BP", "30m_2025年收益%"]
    daily = daily[[c for c in keep_cols_daily if c in daily.columns]].copy()
    m30 = m30[[c for c in keep_cols_m30 if c in m30.columns]].copy()

    out = pd.merge(daily, m30, on="代码", how="outer")
    if "名称_x" in out.columns and "名称_y" in out.columns:
        out["名称"] = out["名称_x"].fillna(out["名称_y"])
        out = out.drop(columns=["名称_x", "名称_y"])
    out = out[["代码", "名称"] + [c for c in out.columns if c not in ("代码", "名称")]]

    # attach data availability (30m)
    if avail is not None and "代码" in avail.columns:
        avail = avail.rename(
            columns={
                "数据源": "30m_数据源",
                "2025可用K线数": "30m_2025可用K线数",
                "数据起始": "30m_数据起始",
                "数据结束": "30m_数据结束",
                "覆盖完整2025": "30m_覆盖完整2025",
            }
        )
        keep_avail = ["代码", "30m_数据源", "30m_2025可用K线数", "30m_数据起始", "30m_数据结束", "30m_覆盖完整2025"]
        out = pd.merge(out, avail[[c for c in keep_avail if c in avail.columns]], on="代码", how="left")

    out = _to_num(
        out,
        [
            "日线_交易次数",
            "日线_胜率%",
            "日线_累计BP",
            "日线_均笔BP",
            "日线_2025年收益%",
            "30m_交易次数",
            "30m_胜率%",
            "30m_累计BP",
            "30m_均笔BP",
            "30m_2025年收益%",
            "30m_2025可用K线数",
            "30m_覆盖完整2025",
        ],
    )

    out["收益差(30m-日线)%"] = out["30m_2025年收益%"] - out["日线_2025年收益%"]

    # sort: prefer symbols with full 30m coverage, then by 30m return
    sort_cols = []
    if "30m_覆盖完整2025" in out.columns:
        sort_cols.append("30m_覆盖完整2025")
    sort_cols.append("30m_2025年收益%")
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    out_dir = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "zhiyue_compare_30m_vs_daily_2025.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved -> {out_path}")
    print(f"daily={daily_sym_csv}")
    print(f"30m={m30_sym_csv}")
    if avail is not None:
        print(f"30m_availability={m30_avail_csv}")


if __name__ == "__main__":
    main()

