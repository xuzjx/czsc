# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def calculate_stats(results_path):
    all_pairs = []
    poss_dir = os.path.join(results_path, "poss")
    if not os.path.exists(poss_dir):
        return None
    
    for symbol in os.listdir(poss_dir):
        symbol_dir = os.path.join(poss_dir, symbol)
        if not os.path.isdir(symbol_dir):
            continue
        
        pairs_file = os.path.join(symbol_dir, "PenZoneStockLongV1.pairs")
        if os.path.exists(pairs_file):
            print(f"Reading {pairs_file}...")
            try:
                df = pd.read_parquet(pairs_file)
                print(f"Read {len(df)} rows from {symbol}")
                all_pairs.append(df)
            except Exception as e:
                print(f"Error reading {pairs_file}: {e}")
    
    if not all_pairs:
        return None
    
    df_all = pd.concat(all_pairs, ignore_index=True)
    if df_all.empty:
        return None
    
    # 计算指标
    # 盈亏比例通常是 (平仓价 - 开仓价) / 开仓价
    # 或者是直接从列 '盈亏比例' 中获取
    returns = df_all['盈亏比例']
    
    total_trades = len(df_all)
    win_rate = len(returns[returns > 0]) / total_trades if total_trades > 0 else 0
    
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    
    avg_win = pos_returns.mean() if not pos_returns.empty else 0
    avg_loss = abs(neg_returns.mean()) if not neg_returns.empty else 1 # avoid div by zero
    profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    # 累计收益 (BP) - 简单加总
    total_return = returns.sum()
    avg_return = returns.mean()
    
    # 假设回测周期是 2 年 (2021-01-01 to 2023-01-01)
    # 年化收益 = (1 + total_return / 标的数) ^ (1/2) - 1  -- 这里简单用平均值代替
    # 或者直接输出总收益率
    
    return {
        "交易次数": total_trades,
        "胜率": f"{win_rate:.2%}",
        "盈亏比": f"{profit_loss_ratio:.2f}",
        "平均收益": f"{avg_return:.4%}",
        "累计总收益": f"{total_return:.4%}"
    }

tags = [
    ("_bt_pen_zone_etf5_20210101_20230101_5m", "包含 ATR 过滤"),
    ("_bt_pen_zone_etf5_no_atr_20210101_20230101_5m", "去掉 ATR 过滤")
]

print("========== 策略收益对比 ==========")
for tag, desc in tags:
    results_path = os.path.join(r"d:\pywork\czsc\examples", tag, "results")
    stats = calculate_stats(results_path)
    print(f"\n模式: {desc}")
    if stats:
        for k, v in stats.items():
            print(f"{k}: {v}")
    else:
        print("无结果或读取失败")
