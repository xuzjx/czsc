import os
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from czsc.utils.plotly_plot import KlineChart

def visualize_portfolio_trades(symbol, root_tag, playback=True):
    # 默认使用仓库下 examples；也支持通过 CZSC_EXAMPLES_ROOT 覆盖
    examples_root = os.environ.get("CZSC_EXAMPLES_ROOT", os.path.dirname(__file__))
    root_path = os.path.join(examples_root, root_tag)
    sigs_file = os.path.join(root_path, "signals", f"{symbol}.sigs")
    weights_file = os.path.join(examples_root, f"{root_tag}_portfolio_top4", "portfolio_weights.parquet")
    
    if not os.path.exists(sigs_file) or not os.path.exists(weights_file):
        print(f"文件缺失: {sigs_file} 或 {weights_file}")
        return

    # 1. 读取数据
    df_sigs = pd.read_parquet(sigs_file)
    df_weights = pd.read_parquet(weights_file)
    
    # 2. 提取该品种的开平仓点
    df_s = df_weights[df_weights['symbol'] == symbol].copy()
    if df_s.empty:
        print(f"品种 {symbol} 在组合回测中没有交易记录")
        return

    df_s['date'] = df_s['dt'].dt.date
    all_dates = sorted(df_s['date'].unique())
    
    entries = []
    exits = []
    
    for i, date in enumerate(all_dates):
        if i == 0 or (date - all_dates[i-1]).days > 4:
            row = df_s[df_s['date'] == date].iloc[0]
            entries.append({'dt': row['dt'], 'price': row['price']})
        
        if i == len(all_dates) - 1 or (all_dates[i+1] - date).days > 4:
            row = df_s[df_s['date'] == date].iloc[0]
            exits.append({'dt': row['dt'], 'price': row['price']})

    df_entries = pd.DataFrame(entries)
    df_exits = pd.DataFrame(exits)

    # 3. 预处理 K 线显示
    df_sigs['dt'] = pd.to_datetime(df_sigs['dt'])
    # 回放模式下，我们取最后 500 根 K 线，演示回放效果
    plot_count = 500 if playback else 2000
    df_plot = df_sigs.tail(plot_count).copy().reset_index(drop=True)
    min_dt = df_plot['dt'].min()
    
    if not df_entries.empty:
        df_entries = df_entries[df_entries['dt'] >= min_dt]
    if not df_exits.empty:
        df_exits = df_exits[df_exits['dt'] >= min_dt]
    
    # 4. 创建基础图表
    kline = KlineChart(n_rows=2, row_heights=(0.8, 0.2), title=f"{symbol} 组合交易回放", width="100%", height=800)
    
    if not playback:
        kline.add_kline(df_plot, name="K线")
        if not df_entries.empty:
            kline.add_scatter_indicator(df_entries['dt'], df_entries['price'], 
                                        name="实际买入", row=1, mode='markers', 
                                        marker_symbol='triangle-up', marker_size=15, marker_color='red')
        if not df_exits.empty:
            kline.add_scatter_indicator(df_exits['dt'], df_exits['price'], 
                                        name="实际卖出", row=1, mode='markers', 
                                        marker_symbol='triangle-down', marker_size=15, marker_color='green')
        output_html = os.path.join(examples_root, f"{symbol}_portfolio_trades.html")
        kline.fig.write_html(output_html)
    else:
        # 回放逻辑：使用 Plotly Frames
        fig = kline.fig
        
        # 初始帧（显示前 100 根）
        initial_count = 100
        df_init = df_plot.iloc[:initial_count]
        
        # 添加基础 K 线 trace (index 0)
        fig.add_trace(go.Candlestick(
            x=df_init['dt'], open=df_init['open'], high=df_init['high'], low=df_init['low'], close=df_init['close'],
            name="K线", increasing_line_color=kline.color_red, decreasing_line_color=kline.color_green
        ), row=1, col=1)
        
        # 添加买入 trace (index 1)
        df_ent_init = df_entries[df_entries['dt'] <= df_init['dt'].max()]
        fig.add_trace(go.Scatter(
            x=df_ent_init['dt'] if not df_ent_init.empty else [None],
            y=df_ent_init['price'] if not df_ent_init.empty else [None],
            name="实际买入", mode='markers', marker=dict(symbol='triangle-up', size=15, color='red')
        ), row=1, col=1)
        
        # 添加卖出 trace (index 2)
        df_ext_init = df_exits[df_exits['dt'] <= df_init['dt'].max()]
        fig.add_trace(go.Scatter(
            x=df_ext_init['dt'] if not df_ext_init.empty else [None],
            y=df_ext_init['price'] if not df_ext_init.empty else [None],
            name="实际卖出", mode='markers', marker=dict(symbol='triangle-down', size=15, color='green')
        ), row=1, col=1)

        # 创建帧
        frames = []
        step = 5 # 每帧增加 5 根 K 线
        for i in range(initial_count, len(df_plot) + 1, step):
            df_curr = df_plot.iloc[:i]
            curr_dt_max = df_curr['dt'].max()
            
            df_ent_curr = df_entries[df_entries['dt'] <= curr_dt_max]
            df_ext_curr = df_exits[df_exits['dt'] <= curr_dt_max]
            
            frames.append(go.Frame(
                data=[
                    go.Candlestick(x=df_curr['dt'], open=df_curr['open'], high=df_curr['high'], low=df_curr['low'], close=df_curr['close']),
                    go.Scatter(x=df_ent_curr['dt'] if not df_ent_curr.empty else [None], y=df_ent_curr['price'] if not df_ent_curr.empty else [None]),
                    go.Scatter(x=df_ext_curr['dt'] if not df_ext_curr.empty else [None], y=df_ext_curr['price'] if not df_ext_curr.empty else [None])
                ],
                name=f"frame_{i}"
            ))
        
        fig.frames = frames
        
        # 添加播放按钮和滑块
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                        "label": "播放 (Play)",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "暂停 (Pause)",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"font": {"size": 20}, "prefix": "进度: ", "visible": True, "xanchor": "right"},
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"frame_{i}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                        "label": str(df_plot.iloc[i-1]['dt'].date()) if i <= len(df_plot) else "",
                        "method": "animate"
                    } for i in range(initial_count, len(df_plot) + 1, step)
                ]
            }]
        )
        
        output_html = os.path.join(r"d:\pywork\czsc\examples", f"{symbol}_playback.html")
        fig.write_html(output_html)
    
    print(f"可视化结果已保存至: {output_html}")
    return output_html

if __name__ == "__main__":
    tag = "_bt_test_visualize"
    visualize_portfolio_trades("159901.SZ", tag, playback=True)
