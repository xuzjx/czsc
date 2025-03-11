from czsc.signals import cxt_15m_zs_3buy_V1


def check():
    from czsc.connectors import research
    from czsc.traders.base import check_signals_acc

    # 获取历史k线数据
    symbols = research.get_symbols('A股主要指数')
    bars = research.get_raw_bars(symbols[0], '1分钟', '20181101', '20210101', fq='前复权')

    signals_config = [{'name': cxt_15m_zs_3buy_V1, 'freq1': "15分钟", 'freq2': "1分钟"}]
    check_signals_acc(bars, signals_config=signals_config, height='780px', delta_days=1)  # type: ignore


if __name__ == '__main__':
    check()