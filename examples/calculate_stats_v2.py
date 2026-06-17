# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def calculate_stats(tag):
    root_path = os.path.join(r"d:\pywork\czsc\examples", tag)
    results_path = os.path.join(root_path, "results")
    poss_dir = os.path.join(results_path, "poss")
    
    if not os.path.exists(poss_dir):
        return None
    
    all_pairs = []
    symbols = [d for d in os.listdir(poss_dir) if os.path.isdir(os.path.join(poss_dir, d))]
    num_symbols = len(symbols)
    
    for symbol in symbols:
        pairs_file = os.path.join(poss_dir, symbol, "PenZoneStockLongV1.pairs")
        if os.path.exists(pairs_file):
            try:
                df = pd.read_parquet(pairs_file)
                all_pairs.append(df)
            except Exception as e:
                pass
    
    if not all_pairs:
        return None
    
    df_all = pd.concat(all_pairs, ignore_index=True)
    returns = df_all['盈亏比例'] # 单位: BP
    
    total_trades = len(df_all)
    win_rate = len(returns[returns > 0]) / total_trades if total_trades > 0 else 0
    
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    avg_win = pos_returns.mean() if not pos_returns.empty else 0
    avg_loss = abs(neg_returns.mean()) if not neg_returns.empty else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    # 组合层面的指标
    # 累计收益 (BP) = 所有交易 BP 之和 / 标的数量 (简单等权)
    total_return_bp = returns.sum() / num_symbols if num_symbols > 0 else 0
    # 年化收益 (简单估算)
    annualized_return = (total_return_bp / 10000) / 2 # 2年
    
    return {
        "标的数量": num_symbols,
        "总交易次数": total_trades,
        "胜率": f"{win_rate:.2%}",
        "盈亏比": f"{profit_loss_ratio:.2f}",
        "每笔平均收益 (BP)": f"{returns.mean():.2f}",
        "组合累计收益 (BP)": f"{total_return_bp:.2f}",
        "估算年化收益": f"{annualized_return:.2%}"
    }

tags = [
    ("_bt_pen_zone_etf5_20210101_20230101_5m", "包含 ATR 过滤 (5只)"),
    ("_bt_pen_zone_etf5_no_atr_20210101_20230101_5m", "去掉 ATR 过滤 (5只)"),
    ("_bt_pen_zone_etf_all_no_atr_20210101_20230101_5m", "包含 ATR 过滤 (全部ETF)")
]

results = []
for tag, desc in tags:
    stats = calculate_stats(tag)
    if stats:
        stats['模式'] = desc
        results.append(stats)

df_results = pd.DataFrame(results)
cols = ['模式', '标的数量', '总交易次数', '胜率', '盈亏比', '每笔平均收益 (BP)', '组合累计收益 (BP)', '估算年化收益']
df_results = df_results[cols]

print("========== 策略收益对比汇总 ==========")
print(df_results.to_string(index=False))

with open(r"d:\pywork\czsc\examples\comparison_final_v2.txt", "w", encoding="utf-8") as f:
    f.write("========== 策略收益对比汇总 ==========\n")
    f.write(df_results.to_string(index=False))
