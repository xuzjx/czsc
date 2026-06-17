# -*- coding: utf-8 -*-
"""
天勤回测/仿真：期货主力连续合约 + PenZoneA1 策略

默认策略：PenZoneA1Strategy（A0/A1 中枢突破，对齐 JSON策略回放_PenZoneA1Futures.py）
  - base_freq=15分钟（可用 SHIPAN_FREQ 覆盖，如 120分钟）, atr_period=14, in_sl=6, adj=F, T+0
可选策略：PenZoneSingleSystemMt5Strategy（SHIPAN_STRATEGY=single_system_mt5）

默认回测窗口：2021-01-01 ~ 2022-01-01（仅 2021 年一年；含 90 天 warmup 自 2020-10-03）。

K 线周期说明：
  - 15分钟（默认）：tqsdk 直接拉 15m K 线，不经 resample。
  - 5分钟（SHIPAN_FREQ=5分钟 / 5m）：tqsdk 直接拉 300s K 线，不经 resample。
  - 120分钟（SHIPAN_FREQ=120分钟 / 120m / 2h）：tqsdk 拉 15m（900s），再用
    czsc.resample_bars(..., base_freq="15分钟") 合成 120分钟，对齐投研/research 的期货 120m 切分
    （tq_connector.get_raw_bars 亦对拉取结果做 resample；此处用 15m 底稿避免 7200s 原生 K 线对齐偏差）。

环境变量：
  SHIPAN_STRATEGY=a1 | single_system_mt5
  SHIPAN_FREQ=120分钟             K 线周期（czsc Freq 值，如 15分钟/120分钟）；别名 120m、2h
  SHIPAN_WATCH_ONLY=1             只打日志不下单
  SHIPAN_VERBOSE=1                pen_zone_a1 信号日志（默认关）
  SHIPAN_ADJ=F|B|N                复权类型，默认 F（前复权）；投研数据为后复权，对齐研究回测可用 B

示例（120m 回测并下单）：
  set SHIPAN_FREQ=120m && set SHIPAN_WATCH_ONLY=0 && python examples/shipan_1.py
"""
import os
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, TargetPosTask, BacktestFinished
import czsc
from czsc import Freq, RawBar
from czsc.signals.strategies_pen_zone_a0 import PenZoneSingleSystemMt5Strategy
from czsc.signals.strategies_pen_zone_a1 import PenZoneA1Strategy

STRATEGY_MAP = {
    "single_system_mt5": PenZoneSingleSystemMt5Strategy,
    "a1": PenZoneA1Strategy,
}

WARMUP_DAYS = 90

# czsc Freq 字符串别名 -> Freq.value（枚举无「2小时」，120 分钟为 F120="120分钟"）
_FREQ_ALIASES = {
    "120m": Freq.F120.value,
    "2h": Freq.F120.value,
    "2小时": Freq.F120.value,
    "5m": Freq.F5.value,
    "15m": Freq.F15.value,
    "30m": Freq.F30.value,
    "60m": Freq.F60.value,
}


def _resolve_shipan_freq(raw: str) -> str:
    """解析 SHIPAN_FREQ / kwargs.freq，返回 czsc Freq 字符串；空串表示用策略类默认。"""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw in _FREQ_ALIASES:
        return _FREQ_ALIASES[raw]
    valid = {f.value for f in Freq}
    if raw in valid:
        return raw
    raise ValueError(
        f"未知 SHIPAN_FREQ={raw!r}，可用: {sorted(v for v in valid if '分钟' in v)} "
        f"或别名 {list(_FREQ_ALIASES)}"
    )


def _freq_to_bar_seconds(freq: str) -> int:
    """tqsdk get_kline_serial 的 duration（秒）。"""
    f = Freq(freq)
    if "分钟" in f.value:
        return int(f.value.replace("分钟", "")) * 60
    if f.value == "日线":
        return 86400
    raise ValueError(f"不支持的周期: {f.value}")


def _needs_15m_resample(freq: str) -> bool:
    """120分钟需从 15m 底稿 resample（tqsdk 不直接拉 7200s 或对齐与 czsc 不一致）。"""
    return Freq(freq) == Freq.F120


def _tq_fetch_seconds(target_freq: str) -> int:
    """tqsdk get_kline_serial duration；120分钟拉 15m 再 resample。"""
    if _needs_15m_resample(target_freq):
        return _freq_to_bar_seconds(Freq.F15.value)
    return _freq_to_bar_seconds(target_freq)


def _kline_df_from_serial(kline, bar_seconds: int, symbol: str) -> pd.DataFrame:
    """已收盘 K 线片段 -> resample_bars 所需 DataFrame（dt 为 K 线结束时刻）。"""
    rows = []
    for row in kline.to_dict("records"):
        rows.append(
            {
                "symbol": symbol,
                "dt": datetime.fromtimestamp(row["datetime"] / 1e9) + timedelta(seconds=bar_seconds),
                "open": row["open"],
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
                "vol": row["volume"],
                "amount": row["volume"] * row["close"],
            }
        )
    return pd.DataFrame(rows)


def _raw_bars_from_closed_kline(closed_kline, symbol: str, target_freq: str, fetch_seconds: int):
    """将已收盘 K 线（无最后一根未完成 bar）转为 target_freq 的 RawBar 列表。"""
    closed_kline = closed_kline[closed_kline["volume"] > 0]
    if closed_kline.empty:
        return []
    if _needs_15m_resample(target_freq):
        df = _kline_df_from_serial(closed_kline, fetch_seconds, symbol)
        return czsc.resample_bars(df, target_freq=target_freq, raw_bars=True, base_freq="15分钟")
    return format_kline(closed_kline, freq=target_freq)


def _pick_strategy():
    name = os.environ.get("SHIPAN_STRATEGY", "a1").strip().lower()
    cls = STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(f"未知 SHIPAN_STRATEGY={name!r}，可选: {list(STRATEGY_MAP)}")
    return cls


def _strategy_with_verbose(base_cls, verbose: bool):
    """子类覆盖 signals_config，控制 pen_zone_a1 log 开关。"""
    if not verbose:
        class _S(base_cls):
            @property
            def signals_config(self):
                return [{**c, "log": False} for c in super().signals_config]

        _S.__name__ = base_cls.__name__
        _S.__qualname__ = base_cls.__qualname__
        return _S
    return base_cls


def _strategy_with_freq(base_cls, freq: str):
    """子类覆盖 base_freq（PenZoneA1 等策略用类属性指定周期）。"""
    if not freq:
        return base_cls

    class _S(base_cls):
        base_freq = freq

    _S.__name__ = base_cls.__name__
    _S.__qualname__ = base_cls.__qualname__
    return _S


def _last_signal_desc(trader) -> str:
    for pos in trader.positions:
        if pos.operates:
            return pos.operates[-1].get("op_desc", "") or pos.operates[-1].get("op", "")
    return ""


def format_kline(df, freq=Freq.F1):
    """对分钟K线进行格式化（dt = K线开盘时刻 + 周期，对齐 tq_connector）"""
    freq = Freq(freq)
    if "分钟" in freq.value:
        duration_seconds = int(freq.value.replace("分钟", "")) * 60
    elif freq.value == "日线":
        duration_seconds = 86400
    else:
        raise ValueError(f"不支持的周期: {freq.value}")

    rows = df.to_dict("records")
    raw_bars = []
    for i, row in enumerate(rows):
        bar = RawBar(
            symbol=row["symbol"],
            id=i,
            freq=freq,
            dt=datetime.fromtimestamp(row["datetime"] / 1e9) + timedelta(seconds=duration_seconds),
            open=row["open"],
            close=row["close"],
            high=row["high"],
            low=row["low"],
            vol=row["volume"],
            amount=row["volume"] * row["close"],
        )
        raw_bars.append(bar)
    return raw_bars


def run_multi_main_symbol(**kwargs):
    """执行多个主力连续合约的策略"""

    tq_user = kwargs.get("tq_user") or os.environ.get("TQ_USER")
    tq_pwd = kwargs.get("tq_pwd") or os.environ.get("TQ_PWD")
    if not tq_user or not tq_pwd:
        raise ValueError(
            "缺少天勤账号信息：请通过参数传入 tq_user/tq_pwd，或设置环境变量 TQ_USER / TQ_PWD"
        )
    init_money = int(kwargs.get("init_money", 1_000_000))
    # 默认仅 2021 年一年（edt 不含）；对比投研全窗口可传 edt=20230101
    sdt = pd.to_datetime(kwargs.get("sdt", "20210101")).date()
    edt = pd.to_datetime(kwargs.get("edt", "20220101")).date()
    watch_only = bool(kwargs.get("watch_only", os.environ.get("SHIPAN_WATCH_ONLY", "0") == "1"))
    # 观察模式强制关闭 pen_zone_a1 debug；SHIPAN_VERBOSE 默认 0
    verbose = (
        False
        if watch_only
        else bool(kwargs.get("verbose", os.environ.get("SHIPAN_VERBOSE", "0") == "1"))
    )
    adj_type = kwargs.get("adj_type", os.environ.get("SHIPAN_ADJ", "F"))  # F=前复权；投研后复权对齐用 B
    shipan_freq = _resolve_shipan_freq(kwargs.get("freq", os.environ.get("SHIPAN_FREQ", "")))

    Strategy = _strategy_with_verbose(
        _strategy_with_freq(_pick_strategy(), shipan_freq),
        verbose=verbose,
    )
    warmup_sdt = sdt - timedelta(days=WARMUP_DAYS)
    freq_label = shipan_freq or getattr(Strategy, "base_freq", "?")
    fetch_label = (
        f"{_tq_fetch_seconds(freq_label)}s->resample {freq_label}"
        if shipan_freq and _needs_15m_resample(freq_label)
        else freq_label
    )
    logger.info(
        f"策略={Strategy.__name__} | freq={freq_label} | kline={fetch_label} | 回测 {sdt}~{edt} | "
        f"warmup_from={warmup_sdt} | watch_only={watch_only} | verbose={verbose} | adj={adj_type}"
    )

    api = TqApi(
        TqSim(init_money),
        web_gui=True,
        auth=TqAuth(tq_user, tq_pwd),
        backtest=TqBacktest(start_dt=warmup_sdt, end_dt=edt),
    )

    symbols = ["KQ.m@SHFE.hc"]
    pos_multiplier = {"KQ.m@SHFE.hc": 1}

    stats = {"total_bars": 0, "pos_changes": 0, "final_pos": 0}
    pos_change_rows = []
    sdt_ts = pd.Timestamp(sdt)
    edt_ts = pd.Timestamp(edt)

    metas = {}
    for symbol in symbols:
        tactic = Strategy(symbol=symbol)
        target_freq = tactic.base_freq
        fetch_seconds = _tq_fetch_seconds(target_freq)
        kline = api.get_kline_serial(symbol, fetch_seconds, data_length=10000, adj_type=adj_type)
        quote = api.get_quote(symbol)

        # 只用已收盘 K 线；120m 时从 15m resample
        raw_bars = _raw_bars_from_closed_kline(kline.iloc[:-1], symbol, target_freq, fetch_seconds)

        # 回测起点之前的历史全部喂入 init_trader（含 2020-12-31 夜盘，避免首笔漂移）
        trader = tactic.init_trader(raw_bars, sdt=sdt)
        stats["total_bars"] += len(raw_bars)
        target_pos = TargetPosTask(api, quote.underlying_symbol)

        init_pos = int(trader.get_ensemble_pos("vote") * pos_multiplier[symbol])
        logger.info(
            f"初始化 {symbol} | end_dt={trader.end_dt} | pos={init_pos} | "
            f"underlying={quote.underlying_symbol} | init_bars={len(raw_bars)}"
        )
        if not watch_only:
            target_pos.set_target_volume(init_pos)

        metas[symbol] = {
            "symbol": symbol,
            "kline": kline,
            "quote": quote,
            "trader": trader,
            "base_freq": target_freq,
            "fetch_seconds": fetch_seconds,
            "target_pos": target_pos,
            "last_pos": init_pos,
        }

    try:
        while api.wait_update():
            for symbol, meta in metas.items():
                kline = meta["kline"]
                quote = meta["quote"]
                trader = meta["trader"]
                target_pos = meta["target_pos"]

                if api.is_changing(quote, "underlying_symbol"):
                    logger.info(f"主力换月：{quote.datetime} - {quote.underlying_symbol}")
                    if not watch_only:
                        target_pos.set_target_volume(0)
                    target_pos = TargetPosTask(api, quote.underlying_symbol)
                    meta["last_pos"] = 0

                if api.is_changing(kline.iloc[-1], "datetime"):
                    closed_kline = kline.iloc[:-1]
                    new_bars = _raw_bars_from_closed_kline(
                        closed_kline.tail(500),
                        symbol,
                        meta["base_freq"],
                        meta["fetch_seconds"],
                    )
                    new_bars = [x for x in new_bars if x.dt > trader.end_dt and x.vol > 0]

                    for bar in new_bars:
                        stats["total_bars"] += 1
                        if bar.dt < sdt_ts:
                            # 回测起点前仅推进 CZSC/信号状态，不触发仓位（对齐投研 init_bar_generator）
                            trader.update_signals(bar)
                            continue

                        trader.update(bar)
                        pos = int(trader.get_ensemble_pos("vote") * pos_multiplier[symbol])
                        if pos != meta["last_pos"]:
                            sig_desc = _last_signal_desc(trader)
                            if not watch_only:
                                pos_names = {p.name: p.pos for p in trader.positions}
                                logger.info(
                                    f"{bar.dt} {symbol} | pos={pos} | positions={pos_names} | "
                                    f"underlying={quote.underlying_symbol}"
                                )
                            logger.warning(
                                f"仓位变动 {meta['last_pos']} -> {pos} @ {bar.dt}"
                                + (f" | signal={sig_desc}" if sig_desc else "")
                            )
                            if sdt_ts <= bar.dt < edt_ts:
                                pos_change_rows.append(
                                    {
                                        "dt": bar.dt,
                                        "symbol": symbol,
                                        "pos_before": meta["last_pos"],
                                        "pos_after": pos,
                                        "signal": sig_desc,
                                        "underlying": quote.underlying_symbol,
                                    }
                                )
                                stats["pos_changes"] += 1
                            meta["last_pos"] = pos
                            if not watch_only:
                                target_pos.set_target_volume(pos)

                meta["target_pos"] = target_pos
                metas[symbol] = meta

    except BacktestFinished:
        logger.info("回测结束")
    except Exception:
        logger.exception("运行异常")
    finally:
        for symbol, meta in metas.items():
            stats["final_pos"] = int(
                meta["trader"].get_ensemble_pos("vote") * pos_multiplier[symbol]
            )
        logger.info(
            f"汇总 | total_bars={stats['total_bars']} | "
            f"pos_changes={stats['pos_changes']} | final_pos={stats['final_pos']}"
        )
        api.close()

    return {"stats": stats, "pos_changes": pos_change_rows}


if __name__ == "__main__":
    run_multi_main_symbol()
