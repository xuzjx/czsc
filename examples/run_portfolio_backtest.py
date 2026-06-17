# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from czsc.traders.weight_backtest import WeightBacktest

# 配置路径
ROOT_TAG = "_bt_test_visualize"
signals_dir = rf"d:\pywork\czsc\examples\{ROOT_TAG}\signals"
results_path = rf"d:\pywork\czsc\examples\{ROOT_TAG}_portfolio_top4"
os.makedirs(results_path, exist_ok=True)

def run_portfolio_backtest():
    all_sigs = []
    print("正在加载信号文件...")
    files = [f for f in os.listdir(signals_dir) if f.endswith(".sigs")]
    
    for file in tqdm(files):
        symbol = file.replace(".sigs", "")
        df = pd.read_parquet(os.path.join(signals_dir, file))
        df['symbol'] = symbol
        # 提取日期和时间
        df['dt'] = pd.to_datetime(df['dt'])
        df['date'] = df['dt'].dt.date
        df['time'] = df['dt'].dt.strftime('%H:%M')
        all_sigs.append(df)
    
    df_all = pd.concat(all_sigs)
    print(f"加载完成，共 {len(df_all)} 条记录")
    
    # 1. 预处理：计算每日涨幅 (14:30价格 / 开盘价格 - 1)
    # 我们假设 09:35 是开盘后的第一根 5分钟 K 线
    print("计算每日涨幅...")
    df_daily_open = df_all[df_all['time'] == '09:35'][['date', 'symbol', 'open']].rename(columns={'open': 'day_open'})
    df_all = df_all.merge(df_daily_open, on=['date', 'symbol'], how='left')
    df_all['daily_gain'] = df_all['close'] / df_all['day_open'] - 1
    
    # 2. 截面选股逻辑
    # 每天 14:30 产生买入信号
    trade_time = '14:30'
    df_trade = df_all[df_all['time'] == trade_time].copy()
    
    # 获取信号列名
    sig_col = [c for c in df_trade.columns if 'PenZoneStockLongV1' in c][0]
    
    dates = sorted(df_trade['date'].unique())
    holdings = set() # 当前持仓
    portfolio_weights = []
    
    print("执行模拟选股...")
    for date in tqdm(dates):
        df_today = df_trade[df_trade['date'] == date]
        
        # A. 必须卖出的：当前持仓中出现“卖出”信号的
        to_sell = set()
        for symbol in holdings:
            row = df_today[df_today['symbol'] == symbol]
            if not row.empty and str(row.iloc[0][sig_col]).startswith('卖出'):
                to_sell.add(symbol)
        
        if to_sell:
            # print(f"{date} 卖出: {to_sell}")
            holdings -= to_sell
        
        # B. 可以买入的：出现“买入”信号的
        candidates = df_today[df_today[sig_col].str.startswith('买入')].copy()
        
        # 排除已经在持仓中的
        candidates = candidates[~candidates['symbol'].isin(holdings)]
        
        if not candidates.empty:
            # 按涨幅排序
            candidates = candidates.sort_values('daily_gain', ascending=False)
            
            # C. 填充空位
            free_slots = 4 - len(holdings)
            if free_slots > 0:
                new_buys = candidates.head(free_slots)['symbol'].tolist()
                # if new_buys:
                #     print(f"{date} 买入: {new_buys}")
                for s in new_buys:
                    holdings.add(s)
        
        # D. 记录今日最终持仓权重
        if holdings:
            weight = 1.0 / len(holdings) if len(holdings) > 0 else 0
            for symbol in holdings:
                # 获取当前价格
                price_row = df_today[df_today['symbol'] == symbol]
                if not price_row.empty:
                    portfolio_weights.append({
                        'dt': pd.Timestamp(date) + pd.Timedelta(hours=15), # 记录在收盘
                        'symbol': symbol,
                        'weight': weight,
                        'price': price_row.iloc[0]['close']
                    })
    
    dfw = pd.DataFrame(portfolio_weights)
    if dfw.empty:
        print("未产生任何交易记录")
        return
    
    # 保存权重记录，供可视化使用
    weights_path = os.path.join(results_path, "portfolio_weights.parquet")
    dfw.to_parquet(weights_path)
    print(f"权重记录已保存至: {weights_path}")

    print(f"生成权重记录 {len(dfw)} 条，开始回测...")
    
    # 3. 使用 WeightBacktest 进行回测
    wb = WeightBacktest(dfw, fee_rate=0.0002)
    wb.report(results_path)
    
    print("\n========== 组合回测绩效评价 ==========")
    for k, v in wb.stats.items():
        print(f"{k}: {v}")
    
    print(f"\n详细结果已保存至: {results_path}")

if __name__ == "__main__":
    run_portfolio_backtest()
