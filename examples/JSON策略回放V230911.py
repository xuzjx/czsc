# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2023/9/11 11:24
describe: CZSC策略单品种回放工具
"""
import os
import sys

# 确保优先使用当前仓库源码（避免误用 site-packages 已安装版本）
_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from czsc.utils.sig import get_zs_seq
# os.environ['czsc_min_bi_len'] = '7'
# os.environ['czsc_bi_change_th'] = '-1'
os.environ['czsc_max_bi_num'] = '20'
# 信号函数所在模块名；默认使用 czsc 自带信号库
# 如需自定义信号库，可在启动前通过环境变量覆盖，例如：
#   set signals_module_name=my_project.signals  (Windows CMD)
#   $env:signals_module_name="my_project.signals" (PowerShell)
os.environ.setdefault('signals_module_name', 'czsc.signals')
os.environ['czsc_research_cache'] = r"D:\CZSC投研数据"  # 本地数据缓存目录
import json
import streamlit as st
import pandas as pd
from copy import deepcopy
from typing import List
from collections import OrderedDict
from czsc import CZSC, Direction, CzscStrategyBase, CzscTrader, KlineChart, Freq, Operate, Position
from czsc.utils import create_single_signal
from czsc.utils.bar_generator import freq_end_time
from czsc.connectors.research import get_symbols, get_raw_bars
from streamlit_extras.mandatory_date_range import date_range_picker

st.set_page_config(layout="wide", page_title="CZSC策略回放", page_icon="🏖️")


def cxt_up_down_signal(c: CZSC, di=1, **kwargs) -> OrderedDict:
    """简单 UpDown 信号：根据倒数第 di 笔方向给出买入 / 卖出 / 其他

    参数模板："{freq}_D{di}_UpDown"

    - 最新笔向上 -> 买入
    - 最新笔向下 -> 卖出
    - 其他 -> 其他
    """
    di = int(di)
    k1, k2, k3 = f"{c.freq.value}_D{di}_UpDown".split("_")

    if not getattr(c, "bi_list", None) or len(c.bi_list) < di:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")

    bi = c.bi_list[-di]
    direction = getattr(bi, "direction", None)
    if direction == Direction.Up:
        v1 = "买入"
    elif direction == Direction.Down:
        v1 = "卖出"
    else:
        v1 = "其他"

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


import logging

# 设置日志记录
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # 确保 czsc.signals 模块的日志也能被捕获
    signals_logger = logging.getLogger('czsc.signals')
    signals_logger.setLevel(logging.INFO)
    if not signals_logger.handlers:
        signals_logger.addHandler(handler)


class JsonStreamStrategy(CzscStrategyBase):
    """读取 streamlit 传入的 json 策略，进行回测"""

    @property
    def positions(self) -> List[Position]:
        """返回当前的持仓策略"""
        json_strategies = self.kwargs.get("json_strategies")
        assert json_strategies, "请在初始化策略时，传入参数 json_strategies"
        positions = []
        for _, pos in json_strategies.items():
            pos["symbol"] = self.symbol
            positions.append(Position.load(pos))
        return positions

    @property
    def signals_config(self):
        """交易信号参数配置（为 UpDown 和 PenZoneV1 自定义信号补充配置）"""
        from czsc.objects import Signal as CzscSignal
        
        # 尝试加载 pen_zone_signal_V1
        try:
            from czsc.signals.strategies_pen_zone import pen_zone_signal_V1
        except ImportError:
            from czsc.signals import pen_zone_signal_V1

        # 尝试加载 pen_zone_stock_long_signal_v1
        try:
            from czsc.signals.strategies_pen_zone_stock_long import pen_zone_stock_long_signal_v1
        except ImportError:
            pen_zone_stock_long_signal_v1 = None

        # 不再调用 super().signals_config，完全手动构建配置
        config = []

        for sig_str in self.unique_signals:
            sig = CzscSignal(sig_str)
            
            # 处理 UpDown 信号
            if sig.k3 == "UpDown":
                freq = sig.k1
                di = 1
                if sig.k2.startswith("D"):
                    try:
                        di = int(sig.k2[1:])
                    except Exception:
                        di = 1
                new_conf = {"name": cxt_up_down_signal, "freq": freq, "di": di}
                if new_conf not in config:
                    config.append(new_conf)

            # 处理 PenZoneV1 相关信号
            if sig.k3 == "PenZoneV1":
                freq = sig.k1
                di = 1
                if sig.k2.startswith("D"):
                    try:
                        di = int(sig.k2[1:])
                    except Exception:
                        di = 1
                new_conf = {"name": pen_zone_signal_V1, "freq": freq, "di": di}
                if new_conf not in config:
                    config.append(new_conf)

            # 处理 PenZoneStockLongV1 相关信号
            if sig.k3 == "PenZoneStockLongV1" and pen_zone_stock_long_signal_v1:
                freq = sig.k1
                di = 1
                if sig.k2.startswith("D"):
                    try:
                        di = int(sig.k2[1:])
                    except Exception:
                        di = 1
                new_conf = {"name": pen_zone_stock_long_signal_v1, "freq": freq, "di": di}
                if new_conf not in config:
                    config.append(new_conf)

        return config

def show_trader(trader: CzscTrader, files):
    if not trader.freqs or not trader.kas or not trader.positions:
        st.error("当前 trader 没有回测数据")
        return

    freqs = trader.freqs
    tabs = st.tabs(freqs + ['回测记录', '策略详情'])

    i = 0
    for freq in freqs:
        with tabs[i]:
            c = trader.kas[freq]
            df = pd.DataFrame(c.bars_raw)
            kline = KlineChart(n_rows=3, row_heights=(0.5, 0.3, 0.2), title='', width="100%", height=600)
            kline.add_kline(df, name="")

            if len(c.bi_list) > 0:
                bi = pd.DataFrame(
                    [{'dt': x.fx_a.dt, "bi": x.fx_a.fx} for x in c.bi_list]
                    + [{'dt': c.bi_list[-1].fx_b.dt, "bi": c.bi_list[-1].fx_b.fx}]
                )
                fx = pd.DataFrame([{'dt': x.dt, "fx": x.fx} for x in c.fx_list])
                kline.add_scatter_indicator(fx['dt'], fx['fx'], name="分型", row=1, line_width=1.2, visible=True)
                kline.add_scatter_indicator(bi['dt'], bi['bi'], name="笔", row=1, line_width=1.5)

            kline.add_sma(df, ma_seq=(5, 20, 120, 240), row=1, visible=False, line_width=1)
            kline.add_vol(df, row=2, line_width=1)
            kline.add_macd(df, row=3, line_width=1)

            for pos in trader.positions:
                bs_df = pd.DataFrame([x for x in pos.operates if x['dt'] >= c.bars_raw[0].dt])
                if not bs_df.empty:
                    bs_df['dt'] = bs_df['dt'].apply(lambda x: freq_end_time(x, Freq(freq)))
                    bs_df['tag'] = bs_df['op'].apply(lambda x: 'triangle-up' if x == Operate.LO else 'triangle-down')
                    bs_df['color'] = bs_df['op'].apply(lambda x: 'red' if x == Operate.LO else 'silver')
                    kline.add_scatter_indicator(
                        bs_df['dt'],
                        bs_df['price'],
                        name=pos.name,
                        text=bs_df['op_desc'],
                        row=1,
                        mode='text+markers',
                        marker_size=15,
                        marker_symbol=bs_df['tag'],
                        marker_color=bs_df['color'],
                    )

            config = {
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                'modeBarButtonsToRemove': [
                    'toggleSpikelines',
                    'select2d',
                    'zoomIn2d',
                    'zoomOut2d',
                    'lasso2d',
                    'autoScale2d',
                    'hoverClosestCartesian',
                    'hoverCompareCartesian',
                ],
            }
            st.plotly_chart(kline.fig, use_container_width=True, config=config)
        i += 1

    with tabs[i]:
        with st.expander("查看所有开平交易记录", expanded=False):
            show_cols = ['策略标记', '交易方向', '盈亏比例', '开仓时间', '平仓时间', '持仓K线数', '事件序列']
            st.dataframe(st.session_state.pos_pairs[show_cols], use_container_width=True, hide_index=True)

        df = pd.DataFrame([x.evaluate() for x in trader.positions])
        st.dataframe(df, use_container_width=True)

        with st.expander("分别查看多头和空头的表现", expanded=False):
            df1 = pd.DataFrame([x.evaluate('多头') for x in trader.positions])
            st.dataframe(df1, use_container_width=True)

            df2 = pd.DataFrame([x.evaluate('空头') for x in trader.positions])
            st.dataframe(df2, use_container_width=True)

    i += 1
    with tabs[i]:
        with st.expander("查看最新信号", expanded=False):
            if len(trader.s):
                s = {k: v for k, v in trader.s.items() if len(k.split('_')) == 3}
                st.write(s)
            else:
                st.warning("当前没有信号配置信息")

        for file in files:
            with st.expander(f"持仓策略配置：{file.name}", expanded=False):
                st.json(json.loads(file.getvalue().decode("utf-8")), expanded=True)

def init_trader(files, symbol, bar_sdt, sdt, edt):
    """初始化回放参数

    :param files: 策略文件
    :param symbol: 交易标的
    :param bar_sdt: 行情开始日期
    :param sdt: 回放开始日期
    :param edt: 回放结束日期
    """
    assert pd.to_datetime(bar_sdt) < pd.to_datetime(sdt) < pd.to_datetime(edt), "回放起止日期设置错误"

    json_strategies = {file.name: json.loads(file.getvalue().decode("utf-8")) for file in files}
    tactic: CzscStrategyBase = JsonStreamStrategy(
        symbol=symbol,
        signals_module_name=os.environ['signals_module_name'],
        json_strategies=json_strategies
    )
    bars = get_raw_bars(symbol, tactic.base_freq, sdt=bar_sdt, edt=edt)
    bg, bars_right = tactic.init_bar_generator(bars, sdt=sdt)
    trader = CzscTrader(bg=bg, positions=deepcopy(tactic.positions), signals_config=deepcopy(tactic.signals_config))

    st.session_state.trader = deepcopy(trader)
    st.session_state.bars_right = deepcopy(bars_right)
    st.session_state.bars_index = 0
    st.session_state.run = False

    # 跑一遍回测，生成持仓记录，用于回放时给人工检查策略一个参考
    for bar in bars_right:
        trader.on_bar(bar)

    assert trader.positions, "当前策略没有持仓记录"
    pairs = [pd.DataFrame(pos.pairs) for pos in trader.positions if pos.pairs]
    st.session_state.pos_pairs = pd.concat(pairs, ignore_index=True)

def main():
    with st.sidebar:
        with st.form(key='my_form_replay'):
            files = st.file_uploader(label='上传策略文件：', type='json', accept_multiple_files=True)
            col1, col2 = st.columns([1, 1])
            symbol = col1.selectbox("选择交易标的：", get_symbols('ALL'), index=0)
            bar_sdt = col2.date_input(label='行情开始日期：', value=pd.to_datetime('2018-01-01'))
            sdt, edt = date_range_picker("回放起止日期", default_start=pd.to_datetime('2019-01-01'), default_end=pd.to_datetime('2022-01-01'))
            submitted = st.form_submit_button(label='设置回放参数')

    if submitted:
        init_trader(files, symbol, bar_sdt, sdt, edt)

    if files and hasattr(st.session_state, 'trader'):
        trader = deepcopy(st.session_state.trader)
        bars_right = deepcopy(st.session_state.bars_right)
        bars_num = len(bars_right)

        c1, c2, c3, c4, c5 = st.columns([5, 5, 5, 5, 25])

        bar_edt = bars_right[st.session_state.bars_index].dt
        target_bar_edt = c5.text_input('行情定位到指定时间：', placeholder=bar_edt.strftime('%Y-%m-%d %H:%M'), key="bar_edt")
        if target_bar_edt:
            target_bar_edt = pd.to_datetime(target_bar_edt)
            for i, bar in enumerate(bars_right):
                if bar.dt >= target_bar_edt:
                    st.session_state.bars_index = i
                    break

        if c1.button('行情播放'):
            st.session_state.run = True
        if c2.button('行情暂停'):
            st.session_state.run = False
        if c3.button('左移一根K线'):
            st.session_state.bars_index -= 1
        if c4.button('右移一根K线'):
            st.session_state.bars_index += 1

        # 约束 bars_index 的范围在 [0, bars_num - 1]
        st.session_state.bars_index = max(0, st.session_state.bars_index)
        st.session_state.bars_index = min(st.session_state.bars_index, bars_num - 1)

        suffix = f"共{bars_num}根K线" if bars_num < 1000 else f"共{bars_num}根K线，回放数据量较大（超过1000根K线），建议缩小回放时间范围"
        st.caption(f"行情播放时间范围：{bars_right[0].dt} - {bars_right[-1].dt}; 当前K线：{bar_edt}；{suffix}")

        if st.session_state.run:
            idx = st.session_state.bars_index
            bars1 = bars_right[0: idx].copy()
            while bars1:
                bar_ = bars1.pop(0)
                trader.on_bar(bar_)

            bars2 = bars_right[idx:].copy()
            with st.empty():
                while bars2:
                    bar_ = bars2.pop(0)
                    trader.on_bar(bar_)
                    show_trader(trader, files)
                    st.session_state.bars_index += 1

        else:
            bars2 = bars_right[: st.session_state.bars_index + 1].copy()
            with st.empty():
                while bars2:
                    bar_ = bars2.pop(0)
                    trader.on_bar(bar_)
                show_trader(trader, files)
    else:
        st.warning("请上传策略文件, 文件格式为 json，配置回放参数")

if __name__ == '__main__':
    main()
