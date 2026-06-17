# -*- coding: utf-8 -*-
import os
import sys
import json
import hashlib
import glob
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List
from copy import deepcopy
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from stqdm import stqdm as tqdm

# CZSC 相关导入
import czsc
from czsc import CZSC, Direction, CzscStrategyBase, CzscTrader, KlineChart, Freq, Operate, Position
from czsc.utils.bar_generator import freq_end_time
from czsc.connectors.research import get_symbols, get_raw_bars
from czsc.utils import create_single_signal
from streamlit_option_menu import option_menu
from streamlit_extras.mandatory_date_range import date_range_picker
from collections import OrderedDict

# 环境配置
os.environ['base_path'] = r"D:\CTA研究"  # 回测结果保存路径
os.environ.setdefault('signals_module_name', 'czsc.signals')

# 确保优先使用当前仓库源码
_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


st.set_page_config(layout="wide", page_title="CZSC策略回放", page_icon="📈")

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

# ======================================================================================================================
# 信号函数定义 (如果导入失败，系统会尝试从这里寻找)
# ======================================================================================================================

def cxt_up_down_signal(c: CZSC, di=1, **kwargs) -> OrderedDict:
    """简单 UpDown 信号"""
    di = int(di)
    k1, k2, k3 = f"{c.freq.value}_D{di}_UpDown".split("_")
    if not getattr(c, "bi_list", None) or len(c.bi_list) < di:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1="其他")
    bi = c.bi_list[-di]
    v1 = "买入" if bi.direction == Direction.Up else "卖出"
    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)


# ======================================================================================================================
# 策略核心类
# ======================================================================================================================

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
        """核心修复：强制通过绝对路径加载自定义信号函数"""
        from czsc.objects import Signal as CzscSignal
        import sys
        import importlib.util

        # 1. 强制设定项目根目录
        project_root = r"D:\pywork\czsc"
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 2. 定位你的信号文件路径
        signal_file_path = os.path.join(project_root, "czsc", "signals", "strategies_pen_zone.py")

        if not os.path.exists(signal_file_path):
            st.error(f"❌ 找不到信号文件：{signal_file_path}")
            return []

        # 3. 动态加载该模块（绕过 site-packages 的干扰）
        try:
            spec = importlib.util.spec_from_file_location("custom_signals", signal_file_path)
            custom_signals = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_signals)

            # 获取具体的函数
            pen_zone_func = getattr(custom_signals, "pen_zone_signal_V1")
            atr_be_func = getattr(custom_signals, "pos_pen_zone_atr_be_V260319")
        except Exception as e:
            st.error(f"❌ 信号函数加载失败: {e}")
            return []

        # 4. 匹配信号字符串与函数
        config = []
        for sig_str in self.unique_signals:
            sig = CzscSignal(sig_str)

            # 匹配 PenZoneV1 (例如: 5分钟_D1_PenZoneV1)
            if "PenZoneV1" in sig.k3:
                freq = sig.k1
                di = int(sig.k2[1:]) if sig.k2.startswith("D") else 1
                new_conf = {"name": pen_zone_func, "freq": freq, "di": di}
                if new_conf not in config:
                    config.append(new_conf)

            # 匹配 ATRBEV1 止损 (例如: 5分钟_PenZoneV1_ATRBEV1)
            if "ATRBEV1" in sig.k3:
                f = sig.k1
                pos_name = sig.k2
                new_conf = {"name": atr_be_func, "f": f, "pos_name": pos_name}
                if new_conf not in config:
                    config.append(new_conf)

        return config
# ======================================================================================================================
# 工具函数
# ======================================================================================================================

def show_trader(trader: CzscTrader, files):
    """
    显示交易者回测结果的交互式图表

    :param trader: CzscTrader 实例
    :param files: 上传的配置文件列表（用于显示策略 JSON）
    """
    if not trader.freqs or not trader.kas or not trader.positions:
        st.error("当前 trader 没有回测数据，请检查数据源或回测周期设置。")
        return

    # 1. 创建 Tabs：每个周期一个 Tab，外加回测记录和策略详情
    freqs = trader.freqs
    tabs = st.tabs(freqs + ['回测记录', '策略详情'])

    for i, freq in enumerate(freqs):
        with tabs[i]:
            c = trader.kas[freq]
            # 确保 K 线时间戳格式统一
            df = pd.DataFrame(c.bars_raw)
            df['dt'] = pd.to_datetime(df['dt'])

            # 初始化 CZSC 标准 K 线图 (3行：K线、成交量、MACD)
            kline = KlineChart(n_rows=3, row_heights=(0.5, 0.3, 0.2), title='', width="100%", height=600)
            kline.add_kline(df, name="K线")

            # 2. 绘制缠论元素：分型与笔
            if len(c.bi_list) > 0:
                bi_data = []
                for x in c.bi_list:
                    bi_data.append({'dt': pd.to_datetime(x.fx_a.dt), "bi": x.fx_a.fx})
                # 加入最后一笔的终点
                bi_data.append({'dt': pd.to_datetime(c.bi_list[-1].fx_b.dt), "bi": c.bi_list[-1].fx_b.fx})

                bi_df = pd.DataFrame(bi_data)
                fx_df = pd.DataFrame([{'dt': pd.to_datetime(x.dt), "fx": x.fx} for x in c.fx_list])

                kline.add_scatter_indicator(fx_df['dt'], fx_df['fx'], name="分型", row=1, line_width=1.2, visible=False)
                kline.add_scatter_indicator(bi_df['dt'], bi_df['bi'], name="笔", row=1, line_width=1.5)

            # 3. 绘制买卖点标记 (关键修复部分)
            for pos in trader.positions:
                # 获取该持仓策略的所有操作记录
                operates = [x for x in pos.operates]
                if not operates:
                    continue

                bs_df = pd.DataFrame(operates)
                # 转换时间戳，确保与 K 线 X 轴完全对齐
                bs_df['dt'] = pd.to_datetime(bs_df['dt'])

                # 定义标记样式
                # LO: 多头开仓, LE: 多头平仓, SO: 空头开仓, SE: 空头平仓
                bs_df['tag'] = bs_df['op'].apply(lambda x: 'triangle-up' if 'O' in x.value else 'triangle-down')
                bs_df['color'] = bs_df['op'].apply(lambda x: 'red' if 'L' in x.value else 'green')

                kline.add_scatter_indicator(
                    bs_df['dt'],
                    bs_df['price'],
                    name=f"{pos.name} 信号",
                    text=bs_df['op_desc'],
                    row=1,
                    mode='markers+text',
                    marker_size=12,
                    marker_symbol=bs_df['tag'],
                    marker_color=bs_df['color'],
                )

            # 4. 辅助指标
            kline.add_sma(df, ma_seq=(5, 20, 120), row=1, visible=True, line_width=1)
            kline.add_vol(df, row=2, line_width=1)
            kline.add_macd(df, row=3, line_width=1)

            # 渲染图表
            st.plotly_chart(kline.fig, use_container_width=True, config={"scrollZoom": True}, key=f"chart_{freq}")

    # 5. 回测记录 Tab
    with tabs[len(freqs)]:
        if hasattr(st.session_state, 'pos_pairs') and not st.session_state.pos_pairs.empty:
            st.subheader("历史交易清单")
            show_cols = ['策略标记', '交易方向', '盈亏比例', '开仓时间', '平仓时间', '持仓K线数', '事件序列']
            st.dataframe(st.session_state.pos_pairs[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("暂无闭合交易记录")

        st.subheader("策略绩效评估")
        eval_df = pd.DataFrame([x.evaluate() for x in trader.positions])
        st.table(eval_df)

    # 6. 策略详情 Tab
    with tabs[len(freqs) + 1]:
        for file in files:
            with st.expander(f"配置文件: {file.name}", expanded=True):
                st.json(json.loads(file.getvalue().decode("utf-8")))

def init_trader(files, symbol, bar_sdt, sdt, edt):
    json_strategies = {file.name: json.loads(file.getvalue().decode("utf-8")) for file in files}
    tactic: CzscStrategyBase = JsonStreamStrategy(
        symbol=symbol, signals_module_name=os.environ['signals_module_name'], json_strategies=json_strategies
    )
    bars = get_raw_bars(symbol, tactic.base_freq, sdt=bar_sdt, edt=edt)
    bg, bars_right = tactic.init_bar_generator(bars, sdt=sdt)
    trader = CzscTrader(bg=bg, positions=deepcopy(tactic.positions), signals_config=deepcopy(tactic.signals_config))

    st.session_state.trader = deepcopy(trader)
    st.session_state.bars_right = deepcopy(bars_right)
    st.session_state.bars_index = 0
    st.session_state.run = False

    for bar in bars_right:
        trader.on_bar(bar)

    assert trader.positions, "当前策略没有持仓记录"
    pairs = [pd.DataFrame(pos.pairs) for pos in trader.positions if pos.pairs]
    if pairs:
        st.session_state.pos_pairs = pd.concat(pairs, ignore_index=True)


def replay(files):
    """CTA策略回放"""
    with st.sidebar:
        with st.form(key='my_form_replay'):
            col1, col2 = st.columns([1, 1])
            symbol = col1.selectbox("选择交易标的：", get_symbols('ALL'), index=0)
            bar_sdt = col2.date_input(label='行情开始日期：', value=pd.to_datetime('2018-01-01'))
            sdt, edt = date_range_picker("回放起止日期", default_start=pd.to_datetime('2019-01-01'),
                                         default_end=pd.to_datetime('2022-01-01'))
            submitted = st.form_submit_button(label='设置回放参数')

    if submitted:
        init_trader(files, symbol, bar_sdt, sdt, edt)

    if files and hasattr(st.session_state, 'trader'):
        trader = deepcopy(st.session_state.trader)
        bars_right = deepcopy(st.session_state.bars_right)
        bars_num = len(bars_right)

        c1, c2, c3, c4, c5 = st.columns([5, 5, 5, 5, 25])
        st.session_state.bars_index = min(st.session_state.bars_index, bars_num - 1)
        bar_edt = bars_right[st.session_state.bars_index].dt

        if c1.button('行情播放'): st.session_state.run = True
        if c2.button('行情暂停'): st.session_state.run = False
        if c3.button('左移'): st.session_state.bars_index = max(0, st.session_state.bars_index - 1)
        if c4.button('右移'): st.session_state.bars_index = min(bars_num - 1, st.session_state.bars_index + 1)

        bars_to_show = bars_right[: st.session_state.bars_index + 1]
        for bar in bars_to_show:
            trader.on_bar(bar)
        show_trader(trader, files)
    else:
        st.warning("请上传 JSON 策略文件并设置回放参数")


# ======================================================================================================================
# 回测代码逻辑 (保持原有逻辑，已确保 JsonStreamStrategy 增强版可被调用)
# ======================================================================================================================

@st.cache_data()
def read_holds_and_pairs(files_traders, pos_name, fee=1):
    holds, pairs = [], []
    for file in tqdm(files_traders):
        try:
            trader = czsc.dill_load(file)
            pos = trader.get_position(pos_name)
            if not pos.holds: continue
            hd = pd.DataFrame(pos.holds)
            hd['symbol'] = trader.symbol
            hd = czsc.subtract_fee(hd, fee=fee)
            holds.append(hd)
            pairs.append(pd.DataFrame(pos.pairs))
        except Exception as e:
            logger.warning(f"{file} 读取失败: {e}")
    return pd.concat(holds, ignore_index=True), pd.concat(pairs, ignore_index=True)


@st.cache_data()
def get_daily_nv(df):
    res = []
    for symbol, hd in df.groupby('symbol'):
        daily = hd.groupby('date').agg({'edge_pre_fee': 'sum', 'edge_post_fee': 'sum'}).reset_index()
        daily['symbol'] = symbol
        res.append(daily)
    return pd.concat(res, ignore_index=True)


def show_backtest_results(file_traders, pos_name, fee=1):
    dfh, dfp = read_holds_and_pairs(file_traders, pos_name, fee=fee)
    dfr = get_daily_nv(dfh)

    st.subheader("一、单笔收益评价")
    pp = czsc.PairsPerformance(dfp)
    df1 = pp.agg_statistics('标的代码')
    st.dataframe(df1, use_container_width=True)

    st.subheader("二、等权收益曲线")
    dfg = dfr.groupby('date').agg({'edge_pre_fee': 'mean', 'edge_post_fee': 'mean'}).cumsum()
    fig = px.line(dfg, title="累计净值曲线")
    st.plotly_chart(fig, use_container_width=True)


def symbol_backtest(strategies, symbol, bar_sdt, sdt, edt, results_path):
    file_trader = results_path / f"{symbol}.trader"
    if file_trader.exists(): return
    try:
        tactic = JsonStreamStrategy(json_strategies=strategies, symbol=symbol)
        bars = get_raw_bars(symbol, tactic.base_freq, sdt=bar_sdt, edt=edt)
        trader = tactic.backtest(bars, sdt=sdt)
        czsc.dill_dump(trader, file_trader)
    except Exception:
        logger.exception(f"{symbol} 回测失败")


@st.cache_data(ttl=3600)
def backtest_all(strategies, results_path, bar_sdt, gruop, sdt, edt, max_workers):
    symbols = get_symbols(gruop)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [executor.submit(symbol_backtest, strategies, symbol, bar_sdt, sdt, edt, results_path) for symbol in
                 symbols]
        for _ in tqdm(as_completed(tasks), total=len(tasks)): pass


def backtest(files):
    with st.sidebar.form(key='bt_form'):
        col1, col2 = st.columns([1, 1])
        gruop = col1.selectbox("回测品类", options=['A股主要指数', 'A股场内基金', '中证500成分股', '期货主力'], index=3)
        bar_sdt = col2.date_input('行情开始日期', value=pd.to_datetime('2018-01-01'))
        sdt, edt = date_range_picker("回放起止日期")
        max_workers = st.number_input('进程数', value=cpu_count() // 4, min_value=1)
        fee = st.number_input('手续费(BP)', value=2)
        submitted = st.form_submit_button('开始回测')

    if submitted:
        strategies = {file.name: json.loads(file.getvalue().decode("utf-8")) for file in files}
        hash_code = hashlib.sha256(f"{str(strategies)}".encode()).hexdigest()[:8].upper()
        results_path = Path(os.getenv("base_path")) / "CTA策略回测" / f"{sdt}_{edt}_{hash_code}" / gruop
        results_path.mkdir(exist_ok=True, parents=True)

        backtest_all(strategies, results_path, bar_sdt, gruop, sdt, edt, max_workers)

        file_traders = glob.glob(fr"{results_path}\*.trader")
        if file_traders:
            trader_one = czsc.dill_load(file_traders[0])
            pos_names = [x.name for x in trader_one.positions]
            pos_name = st.selectbox("选择持仓策略", pos_names)
            show_backtest_results(file_traders, pos_name, fee=fee)


# ======================================================================================================================
# 主入口
# ======================================================================================================================

def main():
    with st.sidebar:
        selected = option_menu("事件驱动择时", ["策略回放", '策略回测'],
                               icons=['play-circle', 'reception-4'], default_index=0)
        files = st.file_uploader('上传策略文件(JSON)', type='json', accept_multiple_files=True)

    if not files:
        st.info("请先上传策略 JSON 文件")
        return

    if selected == "策略回放":
        replay(files)
    else:
        backtest(files)


if __name__ == '__main__':
    main()