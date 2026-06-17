from czsc.signals.strategies_pen_zone_stock_long import pen_zone_stock_long_signal_v1


def check():
    from czsc.connectors import research
    from czsc.traders.base import check_signals_acc

    # 获取历史k线数据
    symbols = research.get_symbols('A股主要指数')
    bars = research.get_raw_bars(symbols[0], '5分钟', '20230101', '20240101', fq='前复权')

    signals_config = [
        {
            'name': pen_zone_stock_long_signal_v1, 
            'freq': "5分钟",
            'di': 1,
            'log': False,
            'trade_time_start': "14:30",
            'trade_time_end': "15:00",
            'ma_period': 250,
            'atr_multiplier': 1.0,
            'inout_count': 1,
        }
    ]
    check_signals_acc(bars, signals_config=signals_config, height='780px', delta_days=5)


if __name__ == '__main__':
    check()