# -*- coding: utf-8 -*-
"""从累计 BP 计算收益率（percent 列均为百分数，如 43.97 表示 43.97%）。

BP 约定：cum_bp / 10000 = 累计收益小数（4431 bp => 44.31%）。

方法说明：
- total_return_pct：单自然年或全区间总收益，不做年化。
- linear_ann_return_pct：线性年化 = cum_bp/10000/years*100（等分各年）。
- compound_ann_return_pct：复利年化 = ((1+cum_bp/10000)^(1/years)-1)*100。
"""

from __future__ import annotations

import numpy as np


def total_return_pct(cum_bp: float) -> float:
    """总收益（%），适用于单自然年。"""
    return float(cum_bp / 10000.0 * 100.0)


def linear_ann_return_pct(cum_bp: float, years: float) -> float:
    """线性年化收益（%）。"""
    if years <= 0:
        return np.nan
    return float(cum_bp / 10000.0 / years * 100.0)


def compound_ann_return_pct(cum_bp: float, years: float) -> float:
    """复利年化收益（%）。"""
    if years <= 0:
        return np.nan
    ratio = 1.0 + cum_bp / 10000.0
    if ratio <= 0:
        return np.nan
    ann = ratio ** (1.0 / years) - 1.0
    return float(ann * 100.0) if np.isfinite(ann) else np.nan
