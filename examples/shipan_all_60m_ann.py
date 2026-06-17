# -*- coding: utf-8 -*-
"""从 shipan_all_60m 汇总表逐品种单跑 TqSdk 回测，提取 TqSim 年化收益。"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd
from loguru import logger
from tqsdk import BacktestFinished, TargetPosTask, TqApi, TqAuth, TqBacktest, TqSim

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.shipan_all_60m import (  # noqa: E402
    BAR_SECONDS,
    BASE_FREQ,
    WARMUP_DAYS,
    _Meta,
    _format_kline,
    _last_signal_desc,
    _strategy_60m,
)
from datetime import timedelta  # noqa: E402


def _parse_tqsdk_report(log_text: str) -> dict:
    """解析 TqSim 回测结束时的绩效行。"""
    m = re.search(
        r"收益率:\s*([-\d.]+)%,\s*年化收益率:\s*([-\d.]+)%,\s*最大回撤:\s*([-\d.]+)%",
        log_text,
    )
    if not m:
        return {}
    return {
        "total_return_pct": float(m.group(1)),
        "ann_return_pct": float(m.group(2)),
        "max_drawdown_pct": float(m.group(3)),
    }


def run_single_symbol_ann(symbol: str, **kwargs) -> dict:
    tq_user = kwargs.get("tq_user", os.environ.get("TQ_USER", "wusuozhu"))
    tq_pwd = kwargs.get("tq_pwd", os.environ.get("TQ_PASS", "1234abcdA!"))
    init_money = int(kwargs.get("init_money", 1_000_000))
    sdt = pd.to_datetime(kwargs.get("sdt", "20210101")).date()
    edt = pd.to_datetime(kwargs.get("edt", "20220101")).date()
    adj_type = kwargs.get("adj_type", "F")
    warmup_sdt = sdt - timedelta(days=WARMUP_DAYS)
    Strategy = _strategy_60m()

    api = TqApi(
        TqSim(init_money),
        web_gui=False,
        auth=TqAuth(tq_user, tq_pwd),
        backtest=TqBacktest(start_dt=warmup_sdt, end_dt=edt),
    )

    tactic = Strategy(symbol=symbol)
    kline = api.get_kline_serial(symbol, BAR_SECONDS, data_length=10000, adj_type=adj_type)
    quote = api.get_quote(symbol)
    hist_kline = kline.iloc[:-1]
    hist_kline = hist_kline[hist_kline["volume"] > 0]
    raw_bars = _format_kline(hist_kline, symbol)
    trader = tactic.init_trader(raw_bars, sdt=sdt)
    target_pos = TargetPosTask(api, quote.underlying_symbol)
    meta = _Meta(
        symbol=symbol,
        kline=kline,
        quote=quote,
        trader=trader,
        target_pos=target_pos,
        last_pos=int(trader.get_ensemble_pos("vote")),
        init_bars=len(raw_bars),
    )
    target_pos.set_target_volume(meta.last_pos)
    sdt_ts = pd.Timestamp(sdt)
    edt_ts = pd.Timestamp(edt)
    pos_changes = 0

    try:
        while api.wait_update():
            if api.is_changing(quote, "underlying_symbol"):
                meta.rollovers += 1
                target_pos.set_target_volume(0)
                target_pos = TargetPosTask(api, quote.underlying_symbol)
                meta.last_pos = 0

            if api.is_changing(kline.iloc[-1], "datetime"):
                closed_kline = kline.iloc[:-1].tail(500)
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
                        if sdt_ts <= bar.dt < edt_ts:
                            pos_changes += 1
                        meta.last_pos = pos
                        target_pos.set_target_volume(pos)
            meta.target_pos = target_pos
    except BacktestFinished:
        pass
    finally:
        account = api.get_account()
        api.close()

    return {
        "symbol": symbol,
        "freq": BASE_FREQ,
        "pos_changes": pos_changes,
        "final_pos": meta.last_pos,
        "total_bars": meta.total_bars,
        "balance": float(account.balance),
        "total_return_pct": float((account.balance / init_money - 1) * 100),
        "ann_return_pct": float(
            ((account.balance / init_money) ** (365.25 / max((edt - sdt).days, 1)) - 1) * 100
        ),
    }


def main():
    sdt = os.environ.get("SHIPAN_SDT", "20210101")
    edt = os.environ.get("SHIPAN_EDT", "20220101")
    init_money = int(os.environ.get("SHIPAN_INIT_MONEY", "1000000"))
    summary_fp = os.path.join(
        _EXAMPLES_DIR, "_portfolio_nav_results", f"shipan_all_60m_summary_{sdt}_{edt}.csv"
    )
    if not os.path.exists(summary_fp):
        raise FileNotFoundError(f"未找到汇总文件: {summary_fp}")

    symbols = pd.read_csv(summary_fp)["symbol"].tolist()
    rows = []
    for sym in symbols:
        logger.info(f"单品种回测年化: {sym} | init_money={init_money}")
        rows.append(run_single_symbol_ann(sym, sdt=sdt, edt=edt, init_money=init_money))

    out = pd.DataFrame(rows).sort_values("ann_return_pct", ascending=False)
    out_fp = os.path.join(
        _EXAMPLES_DIR,
        "_portfolio_nav_results",
        f"shipan_all_60m_ann_{sdt}_{edt}_init{init_money}.csv",
    )
    out.to_csv(out_fp, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n已保存: {out_fp}")


if __name__ == "__main__":
    main()
