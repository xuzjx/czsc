# -*- coding: utf-8 -*-
"""PenZoneA1 回测开平仓点可视化

在 K 线图上标注开多/平多/开空/平空，可选叠加 A0/A1 区间矩形。

启动方式：
    streamlit run examples/PenZoneA1_开平仓可视化.py

或导出静态 HTML（无需启动 Streamlit）：
    py -3 examples/PenZoneA1_开平仓可视化.py --html
    py -3 examples/PenZoneA1_开平仓可视化.py --html --symbol 159901.SZ
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("czsc_max_bi_num", "20")
os.environ.setdefault("czsc_research_cache", r"D:\CZSC投研数据")

DEFAULT_ROOT_TAG = "_bt_pen_zone_a1_stop_etf5_20210101_20230101_15m"
DEFAULT_SYMBOL = "159901.SZ"
POS_NAMES = ("PenZoneA1V1", "PenZoneA1AddV1", "PenZoneA1EtfV1")
TOP_FUTURES_CSV = os.path.join(_EXAMPLES_DIR, "_portfolio_nav_results", "futures_per_symbol_ann_sharpe.csv")

# pairs 文件列名（DummyBacktest 输出）
COL_OPEN_DT = "开仓时间"
COL_CLOSE_DT = "平仓时间"
COL_OPEN_PX = "开仓价格"
COL_CLOSE_PX = "平仓价格"
COL_DIR = "交易方向"

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "toggleSpikelines",
        "select2d",
        "zoomIn2d",
        "zoomOut2d",
        "lasso2d",
        "autoScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


def get_bt_root(root_tag: str = DEFAULT_ROOT_TAG) -> str:
    if os.path.isabs(root_tag):
        return root_tag
    return os.path.join(_EXAMPLES_DIR, root_tag)


def list_bt_roots() -> List[str]:
    """列出本地 PenZoneA1 回测输出目录，优先展示 stop 版本。"""
    roots = []
    for name in os.listdir(_EXAMPLES_DIR):
        path = os.path.join(_EXAMPLES_DIR, name)
        if not os.path.isdir(path) or not name.startswith("_bt_pen_zone_a1"):
            continue
        if os.path.isdir(os.path.join(path, "signals")) and os.path.isdir(os.path.join(path, "results", "poss")):
            roots.append(name)
    return sorted(roots, key=lambda x: (not x.startswith("_bt_pen_zone_a1_stop"), x))


def pick_default_root() -> str:
    roots = list_bt_roots()
    if DEFAULT_ROOT_TAG in roots:
        return DEFAULT_ROOT_TAG
    return roots[0] if roots else DEFAULT_ROOT_TAG


def list_symbols(bt_root: str) -> List[str]:
    poss = os.path.join(bt_root, "results", "poss")
    if not os.path.isdir(poss):
        return []
    return sorted(
        name for name in os.listdir(poss)
        if os.path.isdir(os.path.join(poss, name))
    )


def list_pos_names(bt_root: str, symbol: str) -> List[str]:
    poss = os.path.join(bt_root, "results", "poss", symbol)
    if not os.path.isdir(poss):
        return list(POS_NAMES)
    names = sorted(
        os.path.splitext(name)[0] for name in os.listdir(poss)
        if name.endswith(".pairs")
    )
    return names or list(POS_NAMES)


def infer_freq(root_tag: str):
    """从回测目录名推断 RawBar 频率，无法识别时回退 15分钟。"""
    from czsc import Freq

    name = os.path.basename(root_tag).lower()
    freq_map = {
        "_1m": Freq.F1,
        "_5m": Freq.F5,
        "_15m": Freq.F15,
        "_30m": Freq.F30,
        "_60m": Freq.F60,
        "_120m": Freq.F120,
        "_daily": Freq.D,
    }
    for suffix, freq in freq_map.items():
        if name.endswith(suffix):
            return freq
    return Freq.F15


def infer_timeframe_tag(root_tag: str) -> str:
    """从回测目录名提取周期标签，如 120m / 15m / daily。"""
    name = os.path.basename(root_tag).lower()
    for suffix in ("_120m", "_60m", "_30m", "_15m", "_5m", "_1m", "_daily"):
        if name.endswith(suffix):
            return suffix[1:]
    return "15m"


def is_long_only_pos(pos_name: str) -> bool:
    """ETF T+1 只做多仓位使用 PenZoneA1EtfV1 信号。"""
    return "Etf" in pos_name or "etf" in pos_name


def load_top_futures_symbols(root_tag: str, n: int = 5) -> List[str]:
    """从组合夏普排名 CSV 读取与当前回测周期匹配的 Top-N 期货品种。"""
    if not os.path.isfile(TOP_FUTURES_CSV):
        return []
    tf = infer_timeframe_tag(root_tag)
    try:
        df = pd.read_csv(TOP_FUTURES_CSV)
        sub = df[df["timeframe"] == tf].sort_values("sharpe", ascending=False).head(n)
        return sub["symbol"].astype(str).tolist()
    except Exception:
        return []


def load_kline_df(bt_root: str, symbol: str) -> pd.DataFrame:
    sigs_file = os.path.join(bt_root, "signals", f"{symbol}.sigs")
    if not os.path.exists(sigs_file):
        raise FileNotFoundError(f"信号/K线文件不存在: {sigs_file}")
    df = pd.read_parquet(sigs_file)
    df["dt"] = pd.to_datetime(df["dt"])
    cols = ["dt", "open", "high", "low", "close"]
    if "vol" in df.columns:
        cols.append("vol")
    return df[cols].sort_values("dt").reset_index(drop=True)


def load_pairs_df(bt_root: str, symbol: str, pos_name: str) -> pd.DataFrame:
    pairs_file = os.path.join(bt_root, "results", "poss", symbol, f"{pos_name}.pairs")
    if not os.path.exists(pairs_file):
        return pd.DataFrame()
    df = pd.read_parquet(pairs_file)
    for col in (COL_OPEN_DT, COL_CLOSE_DT):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def filter_by_range(df: pd.DataFrame, sdt: pd.Timestamp, edt: pd.Timestamp, dt_col: str = "dt") -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df[dt_col] >= sdt) & (df[dt_col] <= edt)
    return df.loc[mask].copy()


def pairs_to_trade_points(
    df_pairs: pd.DataFrame,
    sdt: Optional[pd.Timestamp] = None,
    edt: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """将 pairs 拆分为四类开平仓标注点。"""
    empty = pd.DataFrame(columns=["dt", "price", "text"])
    result = {
        "开多": empty.copy(),
        "平多": empty.copy(),
        "开空": empty.copy(),
        "平空": empty.copy(),
    }
    if df_pairs.empty:
        return result

    long_pairs = df_pairs[df_pairs[COL_DIR] == "多头"]
    short_pairs = df_pairs[df_pairs[COL_DIR] == "空头"]

    if not long_pairs.empty:
        result["开多"] = pd.DataFrame({
            "dt": long_pairs[COL_OPEN_DT],
            "price": long_pairs[COL_OPEN_PX],
            "text": long_pairs.get("事件序列", pd.Series(["开多"] * len(long_pairs))).apply(
                lambda x: str(x).split(" -> ")[0] if " -> " in str(x) else "开多"
            ),
        })
        result["平多"] = pd.DataFrame({
            "dt": long_pairs[COL_CLOSE_DT],
            "price": long_pairs[COL_CLOSE_PX],
            "text": long_pairs.get("事件序列", pd.Series(["平多"] * len(long_pairs))).apply(
                lambda x: str(x).split(" -> ")[-1] if " -> " in str(x) else "平多"
            ),
        })

    if not short_pairs.empty:
        result["开空"] = pd.DataFrame({
            "dt": short_pairs[COL_OPEN_DT],
            "price": short_pairs[COL_OPEN_PX],
            "text": short_pairs.get("事件序列", pd.Series(["开空"] * len(short_pairs))).apply(
                lambda x: str(x).split(" -> ")[0] if " -> " in str(x) else "开空"
            ),
        })
        result["平空"] = pd.DataFrame({
            "dt": short_pairs[COL_CLOSE_DT],
            "price": short_pairs[COL_CLOSE_PX],
            "text": short_pairs.get("事件序列", pd.Series(["平空"] * len(short_pairs))).apply(
                lambda x: str(x).split(" -> ")[-1] if " -> " in str(x) else "平空"
            ),
        })

    if sdt is not None and edt is not None:
        for key in result:
            result[key] = filter_by_range(result[key], sdt, edt)

    return result


def _df_to_raw_bars(df: pd.DataFrame, symbol: str, freq=None):
    from czsc import Freq, RawBar

    freq = freq or Freq.F15
    bars = []
    for i, row in df.iterrows():
        bars.append(RawBar(
            symbol=symbol,
            id=int(i),
            dt=row["dt"].to_pydatetime() if hasattr(row["dt"], "to_pydatetime") else row["dt"],
            freq=freq,
            open=float(row["open"]),
            close=float(row["close"]),
            high=float(row["high"]),
            low=float(row["low"]),
            vol=float(row.get("vol", 0) or 0),
            amount=0.0,
        ))
    return bars


def filter_zones_by_range(
    zones: List[dict],
    sdt: Optional[pd.Timestamp],
    edt: Optional[pd.Timestamp],
) -> List[dict]:
    """仅保留与时间窗有交集的 A0/A1 矩形。"""
    if not zones or sdt is None or edt is None:
        return zones
    out = []
    for z in zones:
        z0 = pd.to_datetime(z["begin"])
        z1 = pd.to_datetime(z["end"])
        if z1 >= sdt and z0 <= edt:
            out.append(z)
    return out


def collect_pen_zone_zones(
    df_kline: pd.DataFrame,
    symbol: str,
    freq=None,
    *,
    long_only: bool = False,
    sdt: Optional[pd.Timestamp] = None,
    edt: Optional[pd.Timestamp] = None,
) -> Tuple[List[dict], List[dict]]:
    """重放 PenZoneA1 信号，从 pen_zone_a1_state 重建 A0/A1 区间矩形。

    数据来源：signals/{symbol}.sigs 中的 OHLC 逐 K 重放 pen_zone_a1_signal_v1；
    holds/pairs 不含中枢几何，故不读回测持仓文件。
    - A0：a0_array 中每段「脱离 K 线出生快照」(top/bottom/sdt/edt)
    - A1：has_a1 为真时的 a1_zone（A0_01 价格区间，edt 随最新 A0 延伸）
    """
    from czsc import CZSC
    from czsc.signals.strategies_pen_zone_a1 import pen_zone_a1_signal_v1

    bars = _df_to_raw_bars(df_kline, symbol, freq=freq)
    if len(bars) < 2:
        return [], []

    c = CZSC([bars[0]])
    a0_boxes: List[dict] = []
    a1_by_key: Dict[Tuple, dict] = {}
    seen_a0 = set()

    sig_kw = {"di": 1, "log": False, "long_only": long_only}

    for bar in bars[1:]:
        c.update(bar)
        pen_zone_a1_signal_v1(c, **sig_kw)
        state = c.cache.get("pen_zone_a1_state", {})

        for z in state.get("a0_array", []):
            key = (z.get("sdt"), z.get("top"), z.get("bottom"))
            if key in seen_a0:
                continue
            seen_a0.add(key)
            a0_boxes.append({
                "begin": z["sdt"],
                "end": z.get("edt", z["sdt"]),
                "high": float(z["top"]),
                "low": float(z["bottom"]),
                "label": "A0",
            })

        if state.get("has_a1") and state.get("a1_zone"):
            z = state["a1_zone"]
            key = (z.get("sdt"), z.get("top"), z.get("bottom"))
            edt_z = z.get("edt", z["sdt"])
            prev = a1_by_key.get(key)
            if prev is None or pd.to_datetime(edt_z) > pd.to_datetime(prev["end"]):
                a1_by_key[key] = {
                    "begin": z["sdt"],
                    "end": edt_z,
                    "high": float(z["top"]),
                    "low": float(z["bottom"]),
                    "label": "A1",
                }

    a1_boxes = list(a1_by_key.values())
    return (
        filter_zones_by_range(a0_boxes, sdt, edt),
        filter_zones_by_range(a1_boxes, sdt, edt),
    )


def _build_dt_index(df_kline: pd.DataFrame) -> Tuple[List[pd.Timestamp], Dict[pd.Timestamp, int]]:
    """K 线 dt -> category 轴槽位索引（与 KlineChart 的 type=category 对齐）。"""
    dts = [pd.Timestamp(d) for d in df_kline["dt"]]
    return dts, {d: i for i, d in enumerate(dts)}


def _nearest_bar_index(dt_index: Dict[pd.Timestamp, int], t: pd.Timestamp) -> Optional[int]:
    """将 zone 时间对齐到 K 线槽位（精确匹配或最近一根）。"""
    t = pd.Timestamp(t)
    if t in dt_index:
        return dt_index[t]
    if not dt_index:
        return None
    keys = sorted(dt_index.keys())
    if t <= keys[0]:
        return dt_index[keys[0]]
    if t >= keys[-1]:
        return dt_index[keys[-1]]
    for k in keys:
        if k >= t:
            return dt_index[k]
    return dt_index[keys[-1]]


def _zone_shape_coords(
    z: dict,
    dt_index: Dict[pd.Timestamp, int],
    dts: List[pd.Timestamp],
    n_bars: int,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, float, float]]:
    """将 A0/A1 区间转为与 K 线相同的 category 轴坐标 (x0, x1, y0, y1)。

    KlineChart 的 x 轴为 category，x 须与 Candlestick 的 dt 类别一致（不能用 0/1/2 索引）。
    A0 出生快照 begin==end 时，横向扩成至少一根 K 线宽。
    """
    b = pd.Timestamp(z["begin"])
    e = pd.Timestamp(z["end"])
    bi = _nearest_bar_index(dt_index, b)
    ei = _nearest_bar_index(dt_index, e)
    if bi is None or ei is None:
        return None
    if ei < bi:
        bi, ei = ei, bi
    if bi == ei:
        ei = min(bi + 1, n_bars - 1)
    x0, x1 = dts[bi], dts[ei]
    low, high = float(z["low"]), float(z["high"])
    if low > high:
        low, high = high, low
    return x0, x1, low, high


def _add_zone_rects(
    fig,
    zones: List[dict],
    fillcolor: str,
    dt_index: Dict[pd.Timestamp, int],
    dts: List[pd.Timestamp],
    n_bars: int,
    xref: str = "x1",
    yref: str = "y1",
):
    line_color = (
        fillcolor.replace("0.30", "0.85").replace("0.25", "0.8").replace("0.18", "0.75")
    )
    for z in zones:
        coords = _zone_shape_coords(z, dt_index, dts, n_bars)
        if coords is None:
            continue
        x0, x1, y0, y1 = coords
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            xref=xref,
            yref=yref,
            fillcolor=fillcolor,
            layer="below",
            line=dict(width=1.5, color=line_color),
        )


def build_chart(
    df_kline: pd.DataFrame,
    trade_points: Dict[str, pd.DataFrame],
    *,
    title: str,
    show_a0: bool = True,
    show_a1: bool = True,
    show_bi: bool = False,
    symbol: str = DEFAULT_SYMBOL,
    freq=None,
    long_only: bool = False,
    sdt: Optional[pd.Timestamp] = None,
    edt: Optional[pd.Timestamp] = None,
) -> "KlineChart":
    from czsc import CZSC, KlineChart

    kline = KlineChart(n_rows=1, row_heights=(1.0,), title=title, height=780)
    kline.add_kline(df_kline, name="K线")

    if show_bi and len(df_kline) >= 3:
        bars = _df_to_raw_bars(df_kline, symbol, freq=freq)
        c = CZSC(bars)
        if c.bi_list:
            bi = pd.DataFrame(
                [{"dt": x.fx_a.dt, "bi": x.fx_a.fx} for x in c.bi_list]
                + [{"dt": c.bi_list[-1].fx_b.dt, "bi": c.bi_list[-1].fx_b.fx}]
            )
            kline.add_scatter_indicator(bi["dt"], bi["bi"], name="笔", row=1, line_width=1.2, mode="lines+markers")

    if show_a0 or show_a1:
        dts, dt_index = _build_dt_index(df_kline)
        n_bars = len(df_kline)
        a0_boxes, a1_boxes = collect_pen_zone_zones(
            df_kline, symbol, freq=freq, long_only=long_only, sdt=sdt, edt=edt,
        )
        if show_a0:
            _add_zone_rects(
                kline.fig, a0_boxes, "rgba(0, 120, 255, 0.25)", dt_index, dts, n_bars,
            )
        if show_a1:
            _add_zone_rects(
                kline.fig, a1_boxes, "rgba(255, 140, 0, 0.30)", dt_index, dts, n_bars,
            )

    marker_styles = {
        "开多": dict(mode="markers+text", marker_symbol="triangle-up", marker_color="#F9293E", marker_size=14),
        "平多": dict(mode="markers+text", marker_symbol="triangle-down", marker_color="#FFD700", marker_size=14),
        "开空": dict(mode="markers+text", marker_symbol="triangle-down", marker_color="#00BFFF", marker_size=14),
        "平空": dict(mode="markers+text", marker_symbol="triangle-up", marker_color="#7CFC00", marker_size=14),
    }
    skip_markers = {"开空", "平空"} if long_only else set()
    for name, df_pts in trade_points.items():
        if name in skip_markers:
            continue
        if df_pts.empty:
            continue
        style = marker_styles[name]
        kline.add_scatter_indicator(
            df_pts["dt"],
            df_pts["price"],
            name=name,
            text=df_pts["text"],
            row=1,
            textposition="top center",
            **style,
        )

    return kline


def export_html(
    output_path: str,
    symbol: str = DEFAULT_SYMBOL,
    pos_name: str = "PenZoneA1V1",
    root_tag: str = "",
    sdt: str = "2021-01-01",
    edt: str = "2021-06-30",
    show_a0: bool = True,
    show_a1: bool = True,
) -> str:
    root_tag = root_tag or pick_default_root()
    bt_root = get_bt_root(root_tag)
    freq = infer_freq(root_tag)
    long_only = is_long_only_pos(pos_name)
    df_kline = load_kline_df(bt_root, symbol)
    sdt_ts, edt_ts = pd.to_datetime(sdt), pd.to_datetime(edt)
    df_plot = filter_by_range(df_kline, sdt_ts, edt_ts)

    df_pairs = load_pairs_df(bt_root, symbol, pos_name)
    trade_points = pairs_to_trade_points(df_pairs, sdt_ts, edt_ts)

    title = f"{symbol} | {pos_name} | {sdt[:10]} ~ {edt[:10]}"
    chart = build_chart(
        df_plot,
        trade_points,
        title=title,
        show_a0=show_a0,
        show_a1=show_a1,
        symbol=symbol,
        freq=freq,
        long_only=long_only,
        sdt=sdt_ts,
        edt=edt_ts,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    chart.fig.write_html(output_path)
    return output_path


def _parse_cli_args(argv: List[str]) -> dict:
    args = {
        "html": False,
        "symbol": DEFAULT_SYMBOL,
        "pos": "PenZoneA1V1",
        "sdt": "2021-01-01",
        "edt": "2021-06-30",
        "root": pick_default_root(),
        "out": os.path.join(_EXAMPLES_DIR, f"{DEFAULT_SYMBOL}_penzone_a1_trades.html"),
    }
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--html":
            args["html"] = True
        elif token == "--symbol" and i + 1 < len(argv):
            args["symbol"] = argv[i + 1]
            args["out"] = os.path.join(_EXAMPLES_DIR, f"{argv[i + 1]}_penzone_a1_trades.html")
            i += 1
        elif token == "--pos" and i + 1 < len(argv):
            args["pos"] = argv[i + 1]
            i += 1
        elif token == "--sdt" and i + 1 < len(argv):
            args["sdt"] = argv[i + 1]
            i += 1
        elif token == "--edt" and i + 1 < len(argv):
            args["edt"] = argv[i + 1]
            i += 1
        elif token == "--root" and i + 1 < len(argv):
            args["root"] = argv[i + 1]
            i += 1
        elif token == "--out" and i + 1 < len(argv):
            args["out"] = argv[i + 1]
            i += 1
        i += 1
    return args


def run_streamlit():
    import streamlit as st

    try:
        from streamlit_extras.mandatory_date_range import date_range_picker
        has_date_range_picker = True
    except ImportError:
        has_date_range_picker = False

    st.set_page_config(layout="wide", page_title="PenZoneA1 开平仓可视化", page_icon="📈")

    root_tags = list_bt_roots()
    if not root_tags:
        st.error(f"未找到 PenZoneA1 回测结果目录: {_EXAMPLES_DIR}")
        st.stop()

    with st.sidebar:
        st.header("回放配置")
        default_root = pick_default_root()
        root_tag = st.selectbox(
            "回测目录",
            root_tags,
            index=root_tags.index(default_root) if default_root in root_tags else 0,
        )
        bt_root = get_bt_root(root_tag)
        freq = infer_freq(root_tag)
        symbols = list_symbols(bt_root)
        if not symbols:
            st.error(f"目录中未找到品种持仓结果: {bt_root}")
            st.stop()
        top_syms = load_top_futures_symbols(root_tag, n=5)
        top_syms = [s for s in top_syms if s in symbols]
        if top_syms and "futures" in root_tag:
            st.caption("Top5 夏普品种（同周期）")
            picked = st.radio("快速选择", top_syms, horizontal=True, label_visibility="collapsed")
            default_sym_idx = symbols.index(picked) if picked in symbols else 0
        else:
            default_sym_idx = symbols.index(DEFAULT_SYMBOL) if DEFAULT_SYMBOL in symbols else 0
        symbol = st.selectbox("品种", symbols, index=default_sym_idx)
        pos_names = list_pos_names(bt_root, symbol)
        pos_name = st.selectbox("仓位策略", pos_names, index=0)
        long_only = is_long_only_pos(pos_name)
        if has_date_range_picker:
            sdt, edt = date_range_picker(
                "时间范围",
                default_start=pd.to_datetime("2021-01-01"),
                default_end=pd.to_datetime("2021-06-30"),
            )
        else:
            sdt = st.date_input("开始日期", value=pd.to_datetime("2021-01-01"))
            edt = st.date_input("结束日期", value=pd.to_datetime("2021-06-30"))
        show_a0 = st.checkbox("显示 A0 区间", value=True)
        show_a1 = st.checkbox("显示 A1 区间", value=True)
        show_bi = st.checkbox("显示笔", value=False)
        st.caption(
            f"数据目录: `{root_tag}` | 周期: `{freq.value}`"
            + (" | 只做多 ETF" if long_only else " | 双向期货")
        )
        st.caption("A0/A1 几何：由 signals/*.sigs 的 OHLC 重放 pen_zone_a1_signal_v1 重建")

    sdt_ts = pd.to_datetime(sdt)
    edt_ts = pd.to_datetime(edt) + timedelta(days=1) - timedelta(seconds=1)

    try:
        df_kline = load_kline_df(bt_root, symbol)
        df_plot = filter_by_range(df_kline, sdt_ts, edt_ts)
        df_pairs = load_pairs_df(bt_root, symbol, pos_name)
        df_pairs_range = filter_by_range(
            df_pairs, sdt_ts, edt_ts, dt_col=COL_OPEN_DT
        ) if not df_pairs.empty else df_pairs
        trade_points = pairs_to_trade_points(df_pairs, sdt_ts, edt_ts)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    n_trades = len(df_pairs_range)
    n_open_long = len(trade_points["开多"])
    n_open_short = len(trade_points["开空"])
    a0_cnt, a1_cnt = collect_pen_zone_zones(
        df_plot, symbol, freq=freq, long_only=long_only, sdt=sdt_ts, edt=edt_ts,
    )
    a0_cnt, a1_cnt = len(a0_cnt), len(a1_cnt)

    st.title("PenZoneA1 开平仓点可视化")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("K线根数", len(df_plot))
    c2.metric("交易笔数", n_trades)
    c3.metric("开多点", n_open_long)
    c4.metric("开空点", n_open_short if not long_only else "—")
    c5.metric("A0 区间", a0_cnt)
    c6.metric("A1 区间", a1_cnt)

    title = f"{symbol} | {pos_name} | {sdt_ts.date()} ~ {edt_ts.date()}"
    chart = build_chart(
        df_plot,
        trade_points,
        title=title,
        show_a0=show_a0,
        show_a1=show_a1,
        show_bi=show_bi,
        symbol=symbol,
        freq=freq,
        long_only=long_only,
        sdt=sdt_ts,
        edt=edt_ts,
    )
    st.plotly_chart(chart.fig, use_container_width=True, config=PLOTLY_CONFIG)

    with st.expander("图例说明", expanded=False):
        legend = (
            "- **红色▲ 开多** / **金色▼ 平多**\n"
            + ("" if long_only else "- **青色▼ 开空** / **绿色▲ 平空**\n")
            + "- **蓝色半透明矩形**：A0 区间（K 线脱离当前区间时的单根 K 线高低快照）\n"
            + "- **橙色半透明矩形**：A1 中枢（反向路径与 A0_01 重合后，取 A0_01 价格区间并向最新 A0 延伸）\n"
            + "- **数据来源**：K 线来自 `signals/{symbol}.sigs`；开平仓来自 `results/poss/{symbol}/{pos}.pairs`；"
            + "A0/A1 由 OHLC 重放 `pen_zone_a1_signal_v1` 状态机重建（pairs/holds 不含中枢坐标）"
        )
        st.markdown(legend)

    show_cols = [c for c in [
        "策略标记", COL_DIR, COL_OPEN_DT, COL_CLOSE_DT,
        COL_OPEN_PX, COL_CLOSE_PX, "持仓K线数", "盈亏比例", "事件序列",
    ] if c in df_pairs_range.columns]

    with st.expander("开平仓交易记录", expanded=True):
        if df_pairs_range.empty:
            st.info("所选时间范围内无交易记录")
        else:
            st.dataframe(df_pairs_range[show_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    cli = _parse_cli_args(sys.argv[1:])
    if cli["html"]:
        path = export_html(
            cli["out"],
            symbol=cli["symbol"],
            pos_name=cli["pos"],
            root_tag=cli["root"],
            sdt=cli["sdt"],
            edt=cli["edt"],
        )
        print(f"HTML 已生成: {path}")
    else:
        run_streamlit()
