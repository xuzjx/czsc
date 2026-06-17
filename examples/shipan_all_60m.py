# -*- coding: utf-8 -*-
"""
天勤回测：期货主力连续（KQ.m@...）全品种 + PenZoneA1Strategy（60分钟）

输出：
1) examples/_portfolio_nav_results/shipan_watch_pos_changes_all_60m_<sdt>_<edt>.csv
2) examples/_portfolio_nav_results/shipan_all_60m_summary_<sdt>_<edt>.csv

运行（PowerShell）：
  $env:PYTHONPATH="d:\pywork\czsc"
  $env:SHIPAN_SDT="20210101"
  $env:SHIPAN_EDT="20220101"
  $env:SHIPAN_MAX_SYMBOLS="5"          # 可选：先冒烟跑前 N 个
  $env:SHIPAN_ADJ="F"                  # F/B/N
  python examples/shipan_all_60m.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from loguru import logger
from tqsdk import BacktestFinished, TargetPosTask, TqApi, TqAuth, TqBacktest, TqSim

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import czsc  # noqa: E402
from czsc import Freq, RawBar  # noqa: E402
from czsc.connectors.tq_connector import get_symbols as get_tq_symbols  # noqa: E402
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1Strategy  # noqa: E402


WARMUP_DAYS = 90
BASE_FREQ = Freq.F60.value  # "60分钟"
BAR_SECONDS = 60 * 60


def _format_kline(df: pd.DataFrame, symbol: str) -> List[RawBar]:
    """tqsdk K线序列 -> RawBar（dt 用K线结束时刻）"""
    rows = df.to_dict("records")
    bars: List[RawBar] = []
    for i, row in enumerate(rows):
        bars.append(
            RawBar(
                symbol=symbol,
                id=i,
                freq=Freq(BASE_FREQ),
                dt=datetime.fromtimestamp(row["datetime"] / 1e9) + timedelta(seconds=BAR_SECONDS),
                open=row["open"],
                close=row["close"],
                high=row["high"],
                low=row["low"],
                vol=row["volume"],
                amount=row["volume"] * row["close"],
            )
        )
    return bars


def _last_signal_desc(trader) -> str:
    for pos in trader.positions:
        if pos.operates:
            return pos.operates[-1].get("op_desc", "") or pos.operates[-1].get("op", "")
    return ""


def _strategy_60m():
    class _S(PenZoneA1Strategy):
        base_freq = BASE_FREQ

        @property
        def signals_config(self):
            # 关闭 pen_zone_mt5 的大量 debug 打印，避免全品种/逐品种统计时被日志拖慢
            return [{**c, "log": False} for c in super().signals_config]

    _S.__name__ = PenZoneA1Strategy.__name__
    _S.__qualname__ = PenZoneA1Strategy.__qualname__
    return _S


@dataclass
class _Meta:
    symbol: str
    kline: pd.DataFrame
    quote: object
    trader: object
    target_pos: TargetPosTask
    last_pos: int
    init_bars: int
    rollovers: int = 0
    total_bars: int = 0
    pos_changes: int = 0


def run_all_60m(**kwargs):
    tq_user = kwargs.get("tq_user", os.environ.get("TQ_USER", "wusuozhu"))
    tq_pwd = kwargs.get("tq_pwd", os.environ.get("TQ_PASS", "1234abcdA!"))
    init_money = int(kwargs.get("init_money", os.environ.get("SHIPAN_INIT_MONEY", "1000000")))
    sdt = pd.to_datetime(kwargs.get("sdt", os.environ.get("SHIPAN_SDT", "20210101"))).date()
    edt = pd.to_datetime(kwargs.get("edt", os.environ.get("SHIPAN_EDT", "20220101"))).date()
    max_symbols_raw = kwargs.get("max_symbols", os.environ.get("SHIPAN_MAX_SYMBOLS", "")).strip()
    max_symbols = int(max_symbols_raw) if max_symbols_raw else 0
    adj_type = kwargs.get("adj_type", os.environ.get("SHIPAN_ADJ", "F"))

    warmup_sdt = sdt - timedelta(days=WARMUP_DAYS)
    Strategy = _strategy_60m()

    symbols = list(get_tq_symbols())
    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    logger.info(
        f"策略={Strategy.__name__} | freq={BASE_FREQ} | 回测 {sdt}~{edt} | warmup_from={warmup_sdt} | "
        f"symbols={len(symbols)} | adj={adj_type}"
    )

    api = TqApi(
        TqSim(init_money),
        web_gui=False,
        auth=TqAuth(tq_user, tq_pwd),
        backtest=TqBacktest(start_dt=warmup_sdt, end_dt=edt),
    )

    metas: Dict[str, _Meta] = {}
    pos_change_rows: List[dict] = []
    sdt_ts = pd.Timestamp(sdt)
    edt_ts = pd.Timestamp(edt)

    for symbol in symbols:
        tactic = Strategy(symbol=symbol)
        kline = api.get_kline_serial(symbol, BAR_SECONDS, data_length=10000, adj_type=adj_type)
        quote = api.get_quote(symbol)

        # 只用已收盘 K 线；过滤 0 成交
        hist_kline = kline.iloc[:-1]
        hist_kline = hist_kline[hist_kline["volume"] > 0]
        raw_bars = _format_kline(hist_kline, symbol)
        trader = tactic.init_trader(raw_bars, sdt=sdt)
        target_pos = TargetPosTask(api, quote.underlying_symbol)

        init_pos = int(trader.get_ensemble_pos("vote"))
        target_pos.set_target_volume(init_pos)
        metas[symbol] = _Meta(
            symbol=symbol,
            kline=kline,
            quote=quote,
            trader=trader,
            target_pos=target_pos,
            last_pos=init_pos,
            init_bars=len(raw_bars),
            total_bars=len(raw_bars),
        )
        logger.info(
            f"初始化 {symbol} | end_dt={trader.end_dt} | pos={init_pos} | underlying={quote.underlying_symbol} | "
            f"init_bars={len(raw_bars)}"
        )

    try:
        while api.wait_update():
            for symbol, meta in metas.items():
                kline = meta.kline
                quote = meta.quote
                trader = meta.trader
                target_pos = meta.target_pos

                if api.is_changing(quote, "underlying_symbol"):
                    meta.rollovers += 1
                    logger.info(f"主力换月：{quote.datetime} - {symbol} -> {quote.underlying_symbol}")
                    target_pos.set_target_volume(0)
                    target_pos = TargetPosTask(api, quote.underlying_symbol)
                    meta.last_pos = 0

                if api.is_changing(kline.iloc[-1], "datetime"):
                    closed_kline = kline.iloc[:-1]
                    closed_kline = closed_kline.tail(500)
                    closed_kline = closed_kline[closed_kline["volume"] > 0]
                    new_bars = _format_kline(closed_kline, symbol)
                    new_bars = [x for x in new_bars if x.dt > trader.end_dt and x.vol > 0]

                    for bar in new_bars:
                        meta.total_bars += 1
                        if bar.dt < sdt_ts:
                            trader.update_signals(bar)
                            continue

                        trader.update(bar)
                        pos = int(trader.get_ensemble_pos("vote"))
                        if pos != meta.last_pos:
                            sig_desc = _last_signal_desc(trader)
                            logger.warning(
                                f"仓位变动 {symbol} {meta.last_pos} -> {pos} @ {bar.dt}"
                                + (f" | signal={sig_desc}" if sig_desc else "")
                            )
                            if sdt_ts <= bar.dt < edt_ts:
                                pos_change_rows.append(
                                    {
                                        "dt": bar.dt,
                                        "symbol": symbol,
                                        "pos_before": meta.last_pos,
                                        "pos_after": pos,
                                        "signal": sig_desc,
                                        "underlying": quote.underlying_symbol,
                                    }
                                )
                                meta.pos_changes += 1
                            meta.last_pos = pos
                            target_pos.set_target_volume(pos)

                meta.target_pos = target_pos
                metas[symbol] = meta

    except BacktestFinished:
        logger.info("回测结束")
    except Exception:
        logger.exception("运行异常")
        raise
    finally:
        api.close()

    summary_rows = []
    for symbol, meta in metas.items():
        summary_rows.append(
            {
                "symbol": symbol,
                "init_bars": meta.init_bars,
                "total_bars": meta.total_bars,
                "pos_changes": meta.pos_changes,
                "final_pos": meta.last_pos,
                "rollovers": meta.rollovers,
            }
        )

    return {"pos_changes": pos_change_rows, "summary": summary_rows}


def main():
    result = run_all_60m()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_portfolio_nav_results")
    os.makedirs(out_dir, exist_ok=True)
    sdt = os.environ.get("SHIPAN_SDT", "20210101")
    edt = os.environ.get("SHIPAN_EDT", "20220101")

    pos_csv = os.path.join(out_dir, f"shipan_watch_pos_changes_all_60m_{sdt}_{edt}.csv")
    sum_csv = os.path.join(out_dir, f"shipan_all_60m_summary_{sdt}_{edt}.csv")

    df_pos = pd.DataFrame(result.get("pos_changes", []))
    df_sum = pd.DataFrame(result.get("summary", []))
    df_pos.to_csv(pos_csv, index=False, encoding="utf-8-sig")
    df_sum.to_csv(sum_csv, index=False, encoding="utf-8-sig")

    logger.info(f"已导出仓位变动: {pos_csv} rows={len(df_pos)}")
    logger.info(f"已导出汇总: {sum_csv} rows={len(df_sum)}")
    if not df_sum.empty:
        logger.info(
            f"汇总 | symbols={len(df_sum)} | total_bars={int(df_sum['total_bars'].sum())} | "
            f"pos_changes={int(df_sum['pos_changes'].sum())}"
        )


if __name__ == "__main__":
    main()

