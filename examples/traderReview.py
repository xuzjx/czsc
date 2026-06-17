import os
import sys
import logging

# --- 新增日志配置，确保 INFO 级别的日志能打印到控制台 ---
logging.basicConfig(
    level=logging.INFO,  # 设置最低显示级别为 INFO
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] # 强制输出到控制台
)
# ----------------------------------------------------

import os
import sys

# 确保优先使用当前仓库源码（避免误用 site-packages 已安装版本）
_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import czsc
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from czsc.utils import sorted_freqs
from czsc.connectors.research import get_raw_bars, get_symbols

st.set_page_config(layout="wide")
signals_module = st.sidebar.text_input("信号模块名称：", value="czsc.signals")
parser = czsc.SignalsParser(signals_module=signals_module)

plotly_config = {
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
        'hoverCompareCartesian']}

st.title("信号识别结果观察")

with st.sidebar:
    st.header("信号配置")
    with st.form("my_form"):
        conf = st.text_input("请输入信号：", value="15分钟_D1_PenZoneV1_多信号_任意_任意_0")
        symbol_group = st.selectbox("请选择分组：", ['A股主要指数', 'A股场内基金', '中证500成分股', '期货主力'], index=0)
        symbol = st.selectbox("请选择股票：", get_symbols(symbol_group), index=0)
        freqs = st.multiselect("请选择周期：", sorted_freqs, default=['15分钟', '30分钟', '日线', '周线'])
        freqs = czsc.freqs_sorted(freqs)
        sdt = st.date_input("开始日期：", value=pd.to_datetime('2022-01-01'))
        edt = st.date_input("结束日期：", value=pd.to_datetime('2023-01-01'))
        submit_button = st.form_submit_button(label='提交')


# 获取K线，计算信号
# 注意：BarGenerator 只能从更小周期合成更大周期，不能“反向降采样”；
# 因此基础周期必须 <= 信号所需周期。这里强制使用信号中的周期作为基础周期。
sig_freq = conf.split("_")[0]
if sig_freq not in freqs:
    freqs = czsc.freqs_sorted([sig_freq] + list(freqs))

bars = get_raw_bars(symbol, sig_freq, pd.to_datetime(sdt) - timedelta(days=365 * 3), edt)

# 对 PenZoneV1 这种“手动移植/自定义”的信号：不依赖 SignalsParser 的 docstring 解析，
# 直接用函数构造 signals_config，确保页面可用。
if "PenZoneV1" in conf:
    # conf: {freq}_D{di}_PenZoneV1_{v1}_{v2}_{v3}_{score}
    parts = conf.split("_")
    assert len(parts) == 7, f"信号格式异常：{conf}"
    freq = parts[0]
    k2 = parts[1]  # D{di}
    assert k2.startswith("D"), f"k2 格式异常：{k2}"
    di = int(k2[1:])

    from czsc.signals.strategies_pen_zone import pen_zone_signal_V1

    signals_config = [{"name": pen_zone_signal_V1, "freq": freq, "di": di}]
else:
    signals_config = czsc.get_signals_config([conf], signals_module=signals_module)

sigs = czsc.generate_czsc_signals(bars, signals_config, df=True, sdt=sdt)
#sigs.drop(columns=['freq', 'cache'], inplace=True)
sigs.drop(columns=['freq', 'cache'], inplace=True, errors='ignore')
cols = [x for x in sigs.columns if len(x.split('_')) == 3]
assert len(cols) == 1
sigs['match'] = sigs.apply(czsc.Signal(conf).is_match, axis=1)
sigs['text'] = np.where(sigs['match'], sigs[cols[0]], "")

# 在图中绘制指定需要观察的信号
# ... 前面的代码 ...
sigs['match'] = sigs.apply(czsc.Signal(conf).is_match, axis=1)
sigs['text'] = np.where(sigs['match'], sigs[cols[0]], "")

# --- 新增调试代码：在页面上直接展示该列所有出现过的信号及次数 ---
st.subheader("📊 信号分布统计")
st.write(sigs[cols[0]].value_counts())
# -----------------------------------------------------------

# 在图中绘制指定需要观察的信号
chart = czsc.KlineChart(n_rows=3, height=800)
# ...


chart.add_kline(sigs, freqs[0])
#chart.add_sma(sigs, row=1, ma_seq=(5, 10, 20), visible=True)
#chart.add_vol(sigs, row=2)
#chart.add_macd(sigs, row=3)
df1 = sigs[sigs['text'] != ""][['dt', 'text', 'close', 'low']].copy()
chart.add_scatter_indicator(
    x=df1['dt'], 
    y=df1['low'], 
    row=1, 
    name='信号', 
    mode='markers',
    marker_size=20, 
    marker_color='red', 
    marker_symbol='triangle-up'
)
st.plotly_chart(chart.fig, use_container_width=True, config=plotly_config)