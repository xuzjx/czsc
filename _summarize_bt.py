import os, glob
import pandas as pd
import numpy as np

DIRS = [
    ("1m", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_futures_20210101_20230101_1m"),
    ("5m", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_futures_20210101_20230101_5m"),
    ("15m", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_futures_20210101_20230101_15m"),
    ("120m", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_futures_20210101_20230101_120m"),
    ("日线期货", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_futures_20210101_20230101_daily"),
    ("ETF5-15m", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_etf5_20210101_20230101_15m"),
    ("ETF全-日线", r"d:\pywork\czsc\examples\_bt_pen_zone_a1_stop_etf_all_20210101_20230101_daily"),
]
YEARS = 1.99

def stats(d):
    poss = os.path.join(d, "results", "poss")
    pair_files = glob.glob(os.path.join(poss, "*", "*.pairs"))
    per_sym = []
    all_bp = []
    stops = 0
    trades = 0
    for f in pair_files:
        df = pd.read_parquet(f)
        bp_col = df.columns[-1]
        ev_col = df.columns[8]
        bp = pd.to_numeric(df[bp_col], errors="coerce")
        per_sym.append(bp.sum())
        all_bp.append(bp)
        stops += df[ev_col].astype(str).str.contains("止损", na=False).sum()
        trades += len(df)
    all_bp = pd.concat(all_bp)
    per = np.array(per_sym)
    return dict(
        trades=trades,
        syms=len(pair_files),
        avg_bp=float(all_bp.mean()),
        win=float((all_bp > 0).mean() * 100),
        cum=float(all_bp.sum()),
        stops=int(stops),
        sym_cum_mean=float(per.mean()),
        sym_cum_med=float(np.median(per)),
        sym_pos=float((per > 0).mean() * 100),
        sym_ann_mean=float(per.mean() / 10000 / YEARS * 100),
    )

for name, d in DIRS:
    s = stats(d)
    sp = s["stops"] / s["trades"] * 100
    ann = s["cum"] / 10000 / YEARS * 100
    print(
        f"{name}|{s['syms']}|{s['trades']}|{s['avg_bp']:.2f}|{s['win']:.1f}|{s['cum']:.0f}|{s['stops']}|{sp:.1f}|{ann:.1f}|{s['sym_ann_mean']:.2f}|{s['sym_pos']:.0f}|{s['sym_cum_med']:.0f}"
    )
