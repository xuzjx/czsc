# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# 读取之前的汇总数据
# 我们直接从 log 或者通过 check_bt_results.py 获取的数据来计算
# 这里我们假设截面等权收益是累计收益（单位 BP）

total_bp = -679.465
start_date = pd.to_datetime("2021-01-04")
end_date = pd.to_datetime("2022-12-29")

days = (end_date - start_date).days
years = days / 365.0

total_return = total_bp / 10000.0
annualized_return = (1 + total_return)**(1/years) - 1

print(f"累计收益 (BP): {total_bp}")
print(f"累计收益 (%): {total_return * 100:.2f}%")
print(f"回测年数: {years:.2f}")
print(f"年化收益 (%): {annualized_return * 100:.2f}%")
