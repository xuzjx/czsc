import os
import pandas as pd
from loguru import logger
from datetime import date, datetime, timedelta
from tqsdk import TqApi, TqAuth, TqSim, TqBacktest, TargetPosTask, BacktestFinished
from czsc import Freq, RawBar
from czsc.signals.strategies_pen_zone_stock_long import PenZoneStockLongStrategy as Strategy


def format_kline(df, freq_str):
    """
    格式化 K 线，严格保持与参考文件一致
    """
    freq = Freq(freq_str)
    rows = df.to_dict('records')
    raw_bars = []
    for i, row in enumerate(rows):
        # 统一使用毫秒转秒，加 1 分钟偏移
        dt = datetime.fromtimestamp(row["datetime"] / 1e9) + timedelta(minutes=1)
        bar = RawBar(symbol=row['symbol'], id=i, freq=freq, dt=dt,
                     open=row['open'], close=row['close'], high=row['high'],
                     low=row['low'], vol=row['volume'], amount=row['volume'] * row['close'])
        raw_bars.append(bar)
    return raw_bars


def run_multi_main_symbol(tq_user, tq_pwd, init_money=1000000, sdt="20180101", edt="20181201"):
    sdt_date = pd.to_datetime(sdt).date()
    edt_date = pd.to_datetime(edt).date()

    api = TqApi(TqSim(init_money), web_gui=True, auth=TqAuth(tq_user, tq_pwd),
                backtest=TqBacktest(start_dt=sdt_date, end_dt=edt_date))

    symbols = ["KQ.m@SHFE.hc"]
    pos_multiplier = {"KQ.m@SHFE.hc": 1}

    metas = {}
    for symbol in symbols:
        tactic = Strategy(symbol=symbol)
        kline = api.get_kline_serial(symbol, int(tactic.base_freq.strip('分钟')) * 60, data_length=5000, adj_type='B')
        quote = api.get_quote(symbol)

        # 1. 初始化阶段
        raw_bars = format_kline(kline, freq_str=tactic.base_freq)
        trader = tactic.init_trader(raw_bars, sdt=sdt_date - timedelta(days=30))

        target_pos = TargetPosTask(api, quote.underlying_symbol)
        metas[symbol] = {
            "symbol": symbol,
            "kline": kline,
            "quote": quote,
            "trader": trader,
            "base_freq": tactic.base_freq,
            "target_pos": target_pos,
            "last_pos": 0  # 记录上一次持仓
        }

    logger.info(">>> 策略初始化完成，进入回测循环...")

    try:
        while api.wait_update():
            for symbol, meta in metas.items():
                kline, quote, trader, target_pos = meta["kline"], meta["quote"], meta["trader"], meta["target_pos"]

                # 1. 监控换月
                if api.is_changing(quote, "underlying_symbol"):
                    logger.info(f"主力换月：{quote.underlying_symbol}")
                    target_pos.set_target_volume(0)
                    target_pos = TargetPosTask(api, quote.underlying_symbol)
                    meta["target_pos"] = target_pos

                # 2. 信号更新 (仅在 K 线收盘改变时触发)
                if api.is_changing(kline.iloc[-1], "datetime"):
                    # 取 tail(11) 但排除掉最后一根正在跑的（iloc[-1]）
                    completed_kline = kline.iloc[:-1].tail(10)
                    new_bars = format_kline(completed_kline, freq_str=meta['base_freq'])

                    # 只有当前收盘的 K 线时间 大于 策略记录的最后时间，才更新
                    valid_bars = [x for x in new_bars if x.dt > trader.end_dt]

                    for bar in valid_bars:
                        trader.update(bar)

                        # 获取持仓逻辑
                        pos = trader.get_ensemble_pos('vote')

                        if pos != meta.get("last_pos", 0):
                            # logger.success(f"🚀 动作捕捉 | {bar.dt} | 持仓变动: {meta.get('last_pos', 0)} -> {pos}")
                            meta["last_pos"] = pos

                    # 3. 下单指令发给 TargetPosTask
                    target_pos.set_target_volume(int(trader.get_ensemble_pos('vote') * pos_multiplier[symbol]))

    except BacktestFinished:
        logger.info("回测圆满结束。")
    except Exception as e:
        logger.exception(f"异常: {e}")
    finally:
        api.close()


if __name__ == '__main__':
    tq_user = os.environ.get("TQ_USER")
    tq_pwd = os.environ.get("TQ_PWD")
    if not tq_user or not tq_pwd:
        raise ValueError("请设置环境变量 TQ_USER / TQ_PWD，或在调用时显式传入 tq_user/tq_pwd")
    run_multi_main_symbol(tq_user=tq_user, tq_pwd=tq_pwd, sdt="20180101", edt="20180103")