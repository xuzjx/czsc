# coding: utf-8
"""
使用 pyecharts 定制绘图模块

"""

from pyecharts import options as opts
from pyecharts.charts import HeatMap, Kline, Line, Bar, Scatter, Grid, Boxplot
from pyecharts.commons.utils import JsCode
from typing import List
import numpy as np
from czsc.objects import Operate
from .ta import SMA, MACD


def heat_map(data: List[dict],
             x_label: List[str] = None,
             y_label: List[str] = None,
             title: str = "热力图",
             width: str = "900px",
             height: str = "680px") -> HeatMap:
    """绘制热力图

    :param data: 用于绘制热力图的数据，示例如下
        [{'x': '0hour', 'y': '0day', 'heat': 11},
         {'x': '0hour', 'y': '1day', 'heat': 40},
         {'x': '0hour', 'y': '2day', 'heat': 38},
         {'x': '0hour', 'y': '3day', 'heat': 36},
         {'x': '0hour', 'y': '4day', 'heat': 11}]
    :param x_label: x轴标签
    :param y_label: y轴标签
    :param title: 图表标题
    :param width: 图表宽度
    :param height: 图表高度
    :return: 图表
    """

    value = [[s['x'], s['y'], s['heat']] for s in data]
    heat = [s['heat'] for s in data]

    if not x_label:
        x_label = sorted(list(set([s['x'] for s in data])))

    if not y_label:
        y_label = sorted(list(set([s['y'] for s in data])))

    vis_map_opts = opts.VisualMapOpts(pos_left="90%", pos_top="20%", min_=min(heat), max_=max(heat))
    title_opts = opts.TitleOpts(title=title)
    init_opts = opts.InitOpts(page_title=title, width=width, height=height)
    dz_inside = opts.DataZoomOpts(False, "inside", xaxis_index=[0], range_start=80, range_end=100)
    dz_slider = opts.DataZoomOpts(True, "slider", xaxis_index=[0], pos_top="96%", pos_bottom="0%",
                                  range_start=80, range_end=100)
    legend_opts = opts.LegendOpts(is_show=False)

    hm = HeatMap(init_opts=init_opts)
    hm.add_xaxis(x_label)
    hm.add_yaxis("heat", y_label, value, label_opts=opts.LabelOpts(is_show=True, position="inside"))
    hm.set_global_opts(title_opts=title_opts, visualmap_opts=vis_map_opts, legend_opts=legend_opts,
                       xaxis_opts=opts.AxisOpts(grid_index=0), datazoom_opts=[dz_inside, dz_slider])
    return hm


def kline_pro(kline: List[dict],
              fx: List[dict] = [],
              bi: List[dict] = [],
              xd: List[dict] = [],
              bs: List[dict] = [],
              title: str = "缠中说禅K线分析",
              t_seq: List[int] = [],
              width: str = "1400px",
              height: str = '580px',
              zs_a0: List[dict] = [],
              zs_a1: List[dict] = []) -> Grid:

    """绘制缠中说禅K线分析结果

    :param kline: K线
    :param fx: 分型识别结果
    :param bi: 笔识别结果
        {'dt': Timestamp('2020-11-26 00:00:00'),
          'fx_mark': 'd',
          'start_dt': Timestamp('2020-11-25 00:00:00'),
          'end_dt': Timestamp('2020-11-27 00:00:00'),
          'fx_high': 144.87,
          'fx_low': 138.0,
          'bi': 138.0}
    :param xd: 线段识别结果
    :param bs: 买卖点
    :param title: 图表标题
    :param t_seq: 均线系统
    :param width: 图表宽度
    :param height: 图表高度
    :return: 用Grid组合好的图表
    """
    # 配置项设置
    # ------------------------------------------------------------------------------------------------------------------
    bg_color = "#1f212d"  # 背景
    up_color = "#F9293E"
    down_color = "#00aa3b"

    init_opts = opts.InitOpts(bg_color=bg_color, width=width, height=height, animation_opts=opts.AnimationOpts(False))
    title_opts = opts.TitleOpts(title=title, pos_top="1%",
                                title_textstyle_opts=opts.TextStyleOpts(color=up_color, font_size=20),
                                subtitle_textstyle_opts=opts.TextStyleOpts(color=down_color, font_size=12))

    label_show_opts = opts.LabelOpts(is_show=True)
    label_not_show_opts = opts.LabelOpts(is_show=False)
    legend_not_show_opts = opts.LegendOpts(is_show=False)
    red_item_style = opts.ItemStyleOpts(color=up_color)
    green_item_style = opts.ItemStyleOpts(color=down_color)
    k_style_opts = opts.ItemStyleOpts(color=up_color, color0=down_color, border_color=up_color,
                                      border_color0=down_color, opacity=0.8)

    legend_opts = opts.LegendOpts(is_show=True, pos_top="1%", pos_left="30%", item_width=14, item_height=8,
                                  textstyle_opts=opts.TextStyleOpts(font_size=12, color="#0e99e2"))
    brush_opts = opts.BrushOpts(tool_box=["rect", "polygon", "keep", "clear"],
                                x_axis_index="all", brush_link="all",
                                out_of_brush={"colorAlpha": 0.1}, brush_type="lineX")

    axis_pointer_opts = opts.AxisPointerOpts(is_show=True, link=[{"xAxisIndex": "all"}])

    dz_inside = opts.DataZoomOpts(False, "inside", xaxis_index=[0, 1, 2], range_start=80, range_end=100)
    dz_slider = opts.DataZoomOpts(True, "slider", xaxis_index=[0, 1, 2], pos_top="96%",
                                  pos_bottom="0%", range_start=80, range_end=100)

    yaxis_opts = opts.AxisOpts(is_scale=True, min_="dataMin", max_="dataMax",
                               splitline_opts=opts.SplitLineOpts(is_show=False),
                               axislabel_opts=opts.LabelOpts(color="#c7c7c7", font_size=8, position="inside"))

    grid0_xaxis_opts = opts.AxisOpts(type_="category", grid_index=0, axislabel_opts=label_not_show_opts,
                                     split_number=20, min_="dataMin", max_="dataMax",
                                     is_scale=True, boundary_gap=False,
                                     splitline_opts=opts.SplitLineOpts(is_show=False),
                                     axisline_opts=opts.AxisLineOpts(is_on_zero=False))

    tool_tip_opts = opts.TooltipOpts(
        trigger="axis",
        axis_pointer_type="cross",
        background_color="rgba(245, 245, 245, 0.8)",
        border_width=1,
        border_color="#ccc",
        position=JsCode("""
                    function (pos, params, el, elRect, size) {
    					var obj = {top: 10};
    					obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 30;
    					return obj;
    				}
                    """),
        textstyle_opts=opts.TextStyleOpts(color="#000"),
    )

    # 数据预处理
    # ------------------------------------------------------------------------------------------------------------------
    dts = [x['dt'] for x in kline]
    # k_data = [[x['open'], x['close'], x['low'], x['high']] for x in kline]
    k_data = [opts.CandleStickItem(name=i, value=[x['open'], x['close'], x['low'], x['high']])
              for i, x in enumerate(kline)]

    vol = []
    for i, row in enumerate(kline):
        item_style = red_item_style if row['close'] > row['open'] else green_item_style
        bar = opts.BarItem(name=i, value=row['vol'], itemstyle_opts=item_style, label_opts=label_not_show_opts)
        vol.append(bar)

    close = np.array([x['close'] for x in kline], dtype=np.double)
    diff, dea, macd = MACD(close)
    macd_bar = []
    for i, v in enumerate(macd.tolist()):
        item_style = red_item_style if v > 0 else green_item_style
        bar = opts.BarItem(name=i, value=round(v, 4), itemstyle_opts=item_style,
                           label_opts=label_not_show_opts)
        macd_bar.append(bar)

    diff = diff.round(4)
    dea = dea.round(4)

    # K 线主图
    # ------------------------------------------------------------------------------------------------------------------
    chart_k = Kline()
    chart_k.add_xaxis(xaxis_data=dts)
    chart_k.add_yaxis(series_name="Kline", y_axis=k_data,
                      # itemstyle_opts=k_style_opts
                      itemstyle_opts=opts.ItemStyleOpts(opacity=0)  # 让 K 线完全透明
    )

    chart_k.set_global_opts(
        legend_opts=legend_opts,
        datazoom_opts=[dz_inside, dz_slider],
        yaxis_opts=yaxis_opts,
        tooltip_opts=tool_tip_opts,
        axispointer_opts=axis_pointer_opts,
        brush_opts=brush_opts,
        title_opts=title_opts,
        xaxis_opts=grid0_xaxis_opts
    )

    # 绘制一个横跨整个图表的矩形，Y轴范围在 2600 到 2800 之间
    # Pyecharts 的 markArea 如果只指定 Y 轴，默认会横跨整个 X 轴
    # price_range_mark_area_data = []
    # price_range_mark_area_data.append([
    #     {"yAxis": 2650, "itemStyle": {"color": "rgba(255, 100, 0, 0.2)"}},  # 橙色透明，下边界在2600
    #     {"yAxis": 2660, "label": {"show": True, "formatter": "目标价格区间"}}  # 上边界在2800
    # ])
    #
    # # 创建一个临时的Line图层来承载这个矩形，并叠加到K线主图上
    # # 这种方式对于只指定Y轴的markArea非常有效
    # chart_price_range_area = (
    #     Line()
    #     .add_xaxis(xaxis_data=dts)  # 仍然需要X轴数据，但它的具体值在这里不影响markArea的横跨
    #     .add_yaxis(
    #         series_name="PriceRangeArea",
    #         y_axis=[None] * len(dts),  # 空的y轴数据
    #         is_symbol_show=False,
    #         linestyle_opts=opts.LineStyleOpts(opacity=0),  # 让线不可见
    #         label_opts=label_not_show_opts,
    #         markarea_opts=opts.MarkAreaOpts(
    #             is_silent=True,
    #             data=price_range_mark_area_data
    #         )
    #     )
    #     .set_global_opts(
    #         xaxis_opts=grid0_xaxis_opts,
    #         legend_opts=legend_not_show_opts,
    #         yaxis_opts=yaxis_opts  # 确保Y轴配置兼容
    #     )
    # )
    # chart_k = chart_k.overlap(chart_price_range_area)

    # 加入买卖点 - 多头操作 - 空头操作
    # if bs:
    #     long_opens = {'i': [], 'val': []}
    #     long_exits = {'i': [], 'val': []}
    #     short_opens = {'i': [], 'val': []}
    #     short_exits = {'i': [], 'val': []}
    #
    #     for op in bs:
    #         _dt = op['dt']
    #         _price = round(op['price'], 4)
    #         _info = f"{op['op_desc']} - 价格{_price}"
    #
    #         if op['op'] in [Operate.LO]:
    #             long_opens['i'].append(_dt)
    #             long_opens['val'].append([_price, _info])
    #
    #         if op['op'] in [Operate.LE]:
    #             long_exits['i'].append(_dt)
    #             long_exits['val'].append([_price, _info])
    #
    #         if op['op'] in [Operate.SO]:
    #             short_opens['i'].append(_dt)
    #             short_opens['val'].append([_price, _info])
    #
    #         if op['op'] in [Operate.SE]:
    #             short_exits['i'].append(_dt)
    #             short_exits['val'].append([_price, _info])
    #
    #     chart_lo = (
    #         Scatter().add_xaxis(xaxis_data=long_opens['i']).add_yaxis(
    #             series_name="多头操作",
    #             y_axis=long_opens['val'],
    #             symbol_size=25,
    #             symbol='diamond',
    #             label_opts=opts.LabelOpts(is_show=False),
    #             itemstyle_opts=opts.ItemStyleOpts(color='#ff461f'),
    #             tooltip_opts=opts.TooltipOpts(
    #                 textstyle_opts=opts.TextStyleOpts(font_size=12),
    #                 formatter=JsCode("function (params) {return params.value[2];}")
    #             ),
    #         )
    #     )
    #     chart_le = (
    #         Scatter().add_xaxis(xaxis_data=long_exits['i']).add_yaxis(
    #             series_name="多头操作",
    #             y_axis=long_exits['val'],
    #             symbol_size=25,
    #             symbol='diamond',
    #             label_opts=opts.LabelOpts(is_show=False),
    #             itemstyle_opts=opts.ItemStyleOpts(color='#afdd22'),
    #             tooltip_opts=opts.TooltipOpts(
    #                 textstyle_opts=opts.TextStyleOpts(font_size=12),
    #                 formatter=JsCode("function (params) {return params.value[2];}")
    #             ),
    #         )
    #     )
    #     chart_so = (
    #         Scatter().add_xaxis(xaxis_data=short_opens['i']).add_yaxis(
    #             series_name="空头订单",
    #             y_axis=short_opens['val'],
    #             symbol_size=25,
    #             symbol='triangle',
    #             label_opts=opts.LabelOpts(is_show=False),
    #             itemstyle_opts=opts.ItemStyleOpts(color='#ff461f'),
    #             tooltip_opts=opts.TooltipOpts(
    #                 textstyle_opts=opts.TextStyleOpts(font_size=12),
    #                 formatter=JsCode("function (params) {return params.value[2];}")
    #             ),
    #         )
    #     )
    #     chart_se = (
    #         Scatter().add_xaxis(xaxis_data=short_exits['i']).add_yaxis(
    #             series_name="空头订单",
    #             y_axis=short_exits['val'],
    #             symbol_size=25,
    #             symbol='triangle',
    #             label_opts=opts.LabelOpts(is_show=False),
    #             itemstyle_opts=opts.ItemStyleOpts(color='#afdd22'),
    #             tooltip_opts=opts.TooltipOpts(
    #                 textstyle_opts=opts.TextStyleOpts(font_size=12),
    #                 formatter=JsCode("function (params) {return params.value[2];}")
    #             ),
    #         )
    #     )
    #
    #     chart_k = chart_k.overlap(chart_lo)
    #     chart_k = chart_k.overlap(chart_le)
    #     chart_k = chart_k.overlap(chart_so)
    #     chart_k = chart_k.overlap(chart_se)

    # 均线图
    # ------------------------------------------------------------------------------------------------------------------
    # chart_ma = Line()
    # chart_ma.add_xaxis(xaxis_data=dts)
    # if not t_seq:
    #     t_seq = [5, 13, 21]
    #
    # ma_keys = dict()
    # for t in t_seq:
    #     ma_keys[f"MA{t}"] = SMA(close, timeperiod=t)
    #
    # for i, (name, ma) in enumerate(ma_keys.items()):
    #     chart_ma.add_yaxis(series_name=name, y_axis=ma, is_smooth=True,
    #                        symbol_size=0, label_opts=label_not_show_opts,
    #                        linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1))
    #
    # chart_ma.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
    # chart_k = chart_k.overlap(chart_ma)

    # 缠论结果
    # ------------------------------------------------------------------------------------------------------------------
    # if fx:
    #     fx_dts = [x['dt'] for x in fx]
    #     fx_val = [round(x['fx'], 2) for x in fx]
    #     chart_fx = Line()
    #     chart_fx.add_xaxis(fx_dts)
    #     chart_fx.add_yaxis(series_name="FX", y_axis=fx_val,
    #                        symbol="circle", symbol_size=6, label_opts=label_show_opts,
    #                        itemstyle_opts=opts.ItemStyleOpts(color="rgba(152, 147, 193, 1.0)", ))
    #
    #     chart_fx.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
    #     chart_k = chart_k.overlap(chart_fx)

    if bi:
        bi_dts = [x['dt'] for x in bi]
        bi_val = [round(x['bi'], 2) for x in bi]
        chart_bi = Line()
        chart_bi.add_xaxis(bi_dts)
        chart_bi.add_yaxis(series_name="BI", y_axis=bi_val,
                           symbol="diamond", symbol_size=10, label_opts=label_show_opts,
                           itemstyle_opts=opts.ItemStyleOpts(color="rgba(184, 117, 225, 1.0)", ),
                           linestyle_opts=opts.LineStyleOpts(width=1.5))

        chart_bi.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
        chart_k = chart_k.overlap(chart_bi)

    # if xd:
    #     xd_dts = [x['dt'] for x in xd]
    #     xd_val = [x['xd'] for x in xd]
    #     chart_xd = Line()
    #     chart_xd.add_xaxis(xd_dts)
    #     chart_xd.add_yaxis(series_name="XD", y_axis=xd_val,
    #                        symbol="triangle", symbol_size=10,
    #                        itemstyle_opts=opts.ItemStyleOpts(color="rgba(37, 141, 54, 1.0)", ))
    #
    #     chart_xd.set_global_opts(xaxis_opts=grid0_xaxis_opts, legend_opts=legend_not_show_opts)
    #     chart_k = chart_k.overlap(chart_xd)

        # 缠论结果 - 中枢 (ZS) - 使用成功的方法绘制矩形
        # ------------------------------------------------------------------------------------------------------------------
    if zs_a0:
        zs_mark_areas = []

        for i, z in enumerate(zs_a0):
            # 调试打印：检查当前中枢的原始 begin/end 值和类型
            # print(f"\n处理第 {i} 个中枢：")
            # print(f"  原始 z['begin']: {z.get('begin')}, 类型: {type(z.get('begin'))}")
            # print(f"  原始 z['end']: {z.get('end')}, 类型: {type(z.get('end'))}")

            try:
                # 尝试获取 begin 和 end 在 dts 列表中的索引
                # 这里的假设是 z["begin"] 和 z["end"] 的值类型与 dts 中的元素是可比较的
                begin_idx = -1
                end_idx = -1

                # 遍历 dts 列表查找匹配项，而不是直接使用 dts.index()，因为 .index() 要求精确匹配
                # 这种方式更健壮，可以处理一些微妙的类型或格式差异
                for idx, dt_val in enumerate(dts):
                    if dt_val == z["begin"]:
                        begin_idx = idx
                        break
                for idx, dt_val in enumerate(dts):
                    if dt_val == z["end"]:
                        end_idx = idx
                        break

                if begin_idx == -1 or end_idx == -1:
                    print(f"  警告：中枢日期 {z.get('begin')} 或 {z.get('end')} 未在 K线 dts 列表中精确匹配到，已跳过此中枢。")
                    continue

                # 获取中枢高点和低点
                # 如果你的 zs 数据结构里有 high 和 low 键，请优先使用它们
                # 假设 zs 结构为 {'begin': dt_obj, 'end': dt_obj, 'label': '中枢1', 'high': price_high, 'low': price_low}
                # 如果没有，可以使用之前模拟的逻辑，但强烈建议 zs 数据自带价格范围

                zs_high = z.get('high', z.get('zg'))  # 优先取 'high'，其次 'zg'
                zs_low = z.get('low', z.get('zd'))  # 优先取 'low'，其次 'zd'

                # 确保 yAxis 值的顺序是 low 到 high
                if zs_low > zs_high:
                    zs_low, zs_high = zs_high, zs_low

                # 使用 X 轴日期对应的索引作为 xAxis 值
                zs_mark_areas.append([
                    {"xAxis": dts[begin_idx], "yAxis": zs_low, "itemStyle": {"color": "rgba(0, 100, 255, 0.15)"}},
                    {"xAxis": dts[end_idx], "yAxis": zs_high, "label": {"show": True, "formatter": z.get("label", "中枢")}}
                ])
                # print(f"  成功添加中枢矩形数据: X范围 [{dts[begin_idx]}, {dts[end_idx]}] Y范围 [{zs_low}, {zs_high}]")

            except KeyError as e:
                print(f"  错误：中枢数据 {z} 缺少必要的键 {e}，跳过此中枢绘制。")
                continue
            except Exception as e:
                print(f"  绘制中枢矩形时发生未知错误：{e}，跳过此中枢绘制。")
                continue

        # 调试打印：检查最终的 zs_mark_areas 列表
        # print("\n最终 zs_mark_areas 列表：")
        # for item in zs_mark_areas:
            # print(f"  {item}")

        if zs_mark_areas:
            chart_zs_area = (
                Line()
                .add_xaxis(xaxis_data=dts)
                .add_yaxis(
                    series_name="ZS_Area",
                    y_axis=[None] * len(dts),
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(opacity=0),
                    label_opts=label_not_show_opts,
                    markarea_opts=opts.MarkAreaOpts(
                        is_silent=True,
                        data=zs_mark_areas
                    )
                )
                .set_global_opts(
                    xaxis_opts=grid0_xaxis_opts,
                    legend_opts=legend_not_show_opts,
                    yaxis_opts=yaxis_opts
                )
            )
            chart_k = chart_k.overlap(chart_zs_area)
        else:
            print("zs_mark_areas 为空，未添加中枢矩形图层。")

    # ----------------------------------------------------------------------
    # 绘制 A1 级别中枢 (参考 A0 的逻辑进行)
    # ----------------------------------------------------------------------
    # if zs_a1:
    #     zs_mark_areas_a1 = []
    #
    #     print("\n--- 处理 A1 级别中枢 ---")
    #     for i, z in enumerate(zs_a1):
    #         print(f"\n处理第 {i} 个 A1 中枢：")
    #         print(f"  原始 z['begin']: {z.get('begin')}, 类型: {type(z.get('begin'))}")
    #         print(f"  原始 z['end']: {z.get('end')}, 类型: {type(z.get('end'))}")
    #
    #         try:
    #             begin_dt = z["begin"]
    #             end_dt = z["end"]
    #
    #             begin_idx = -1
    #             end_idx = -1
    #             for idx, dt_val in enumerate(dts):
    #                 if dt_val == begin_dt:
    #                     begin_idx = idx
    #                 if dt_val == end_dt:
    #                     end_idx = idx
    #                 if begin_idx != -1 and end_idx != -1:
    #                     break
    #
    #             if begin_idx == -1 or end_idx == -1:
    #                 print(f"  警告：A1 中枢日期 {begin_dt} 或 {end_dt} 未在 K线 dts 列表中精确匹配到，已跳过此中枢。")
    #                 continue
    #
    #             zs_high = z.get('high', z.get('zg'))
    #             zs_low = z.get('low', z.get('zd'))
    #
    #             if zs_low > zs_high:
    #                 zs_low, zs_high = zs_high, zs_low
    #
    #             zs_mark_areas_a1.append([
    #                 {"xAxis": dts[begin_idx], "yAxis": zs_low, "itemStyle": {"color": "rgba(0, 0, 255, 0.15)"}},
    #                 # 蓝色半透明
    #                 {"xAxis": dts[end_idx], "yAxis": zs_high,
    #                  "label": {"show": True, "formatter": z.get("label", f"A1_ZS{i}")}}
    #             ])
    #             print(f"  成功添加 A1 中枢矩形数据: X范围 [{dts[begin_idx]}, {dts[end_idx]}] Y范围 [{zs_low}, {zs_high}]")
    #
    #         except KeyError as e:
    #             print(f"  错误：A1 中枢数据 {z} 缺少必要的键 {e}，跳过此中枢绘制。")
    #             continue
    #         except Exception as e:
    #             print(f"  绘制 A1 中枢矩形时发生未知错误：{e}，跳过此中枢绘制。")
    #             continue
    #
    #     print("\n最终 A1 zs_mark_areas 列表：")
    #     for item in zs_mark_areas_a1:
    #         print(f"  {item}")
    #
    #     if zs_mark_areas_a1:
    #         chart_zs_area_a1 = (
    #             Line()
    #             .add_xaxis(xaxis_data=dts)
    #             .add_yaxis(
    #                 series_name="A1_ZS_Area",
    #                 y_axis=[None] * len(dts),
    #                 is_symbol_show=False,
    #                 linestyle_opts=opts.LineStyleOpts(opacity=0),
    #                 label_opts=opts.LabelOpts(is_show=False),
    #                 markarea_opts=opts.MarkAreaOpts(
    #                     is_silent=True,
    #                     data=zs_mark_areas_a1
    #                 )
    #             )
    #             .set_global_opts(
    #                 xaxis_opts=grid0_xaxis_opts,
    #                 legend_opts=legend_not_show_opts,
    #                 yaxis_opts=yaxis_opts
    #             )
    #         )
    #         chart_k = chart_k.overlap(chart_zs_area_a1)
    #     else:
    #         print("zs_mark_areas_a1 为空，未添加 A1 级别中枢矩形图层。")




    # 成交量图
    # ------------------------------------------------------------------------------------------------------------------
    chart_vol = Bar()
    chart_vol.add_xaxis(dts)
    chart_vol.add_yaxis(series_name="Volume", y_axis=vol, bar_width='60%')
    chart_vol.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=1,
            boundary_gap=False,
            axislabel_opts=opts.LabelOpts(is_show=True, font_size=8, color="#9b9da9"),
        ),
        yaxis_opts=yaxis_opts, legend_opts=legend_not_show_opts,
    )

    # MACD图
    # ------------------------------------------------------------------------------------------------------------------
    chart_macd = Bar()
    chart_macd.add_xaxis(dts)
    chart_macd.add_yaxis(series_name="MACD", y_axis=macd_bar, bar_width='60%')
    chart_macd.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=2,
            axislabel_opts=opts.LabelOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            grid_index=2,
            split_number=4,
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=True, color="#c7c7c7"),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )

    line = Line()
    line.add_xaxis(dts)
    line.add_yaxis(series_name="DIFF", y_axis=diff, label_opts=label_not_show_opts, is_symbol_show=False,
                   linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color="#da6ee8"))
    line.add_yaxis(series_name="DEA", y_axis=dea, label_opts=label_not_show_opts, is_symbol_show=False,
                   linestyle_opts=opts.LineStyleOpts(opacity=0.8, width=1.0, color="#39afe6"))

    chart_macd = chart_macd.overlap(line)

    grid0_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="12%", height="58%")
    grid1_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="74%", height="8%")
    grid2_opts = opts.GridOpts(pos_left="0%", pos_right="1%", pos_top="86%", height="10%")

    grid_chart = Grid(init_opts)
    grid_chart.add(chart_k, grid_opts=grid0_opts)
    grid_chart.add(chart_vol, grid_opts=grid1_opts)
    grid_chart.add(chart_macd, grid_opts=grid2_opts)
    return grid_chart


def box_plot(data: dict,
             title: str = "箱线图",
             width: str = "900px",
             height: str = "680px") -> Boxplot:
    """

    :param data: 数据
        样例：
        data = {
            "expr 0": [960, 850, 830, 880],
            "expr 1": [960, 850, 830, 880],
        }
    :param title:
    :param width:
    :param height:
    :return:
    """
    x_data = []
    y_data = []
    for k, v in data.items():
        x_data.append(k)
        y_data.append(v)

    init_opts = opts.InitOpts(page_title=title, width=width, height=height)

    chart = Boxplot(init_opts=init_opts)
    chart.add_xaxis(xaxis_data=x_data)
    chart.add_yaxis(series_name="", y_axis=y_data)
    chart.set_global_opts(title_opts=opts.TitleOpts(pos_left="center", title=title),
                          tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type="shadow"),
                          xaxis_opts=opts.AxisOpts(
                              type_="category",
                              boundary_gap=True,
                              splitarea_opts=opts.SplitAreaOpts(is_show=False),
                              axislabel_opts=opts.LabelOpts(formatter="{value}"),
                              splitline_opts=opts.SplitLineOpts(is_show=False),
                          ),
                          yaxis_opts=opts.AxisOpts(
                              type_="value",
                              name="",
                              splitarea_opts=opts.SplitAreaOpts(
                                  is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                              )
                          ))
    return chart
