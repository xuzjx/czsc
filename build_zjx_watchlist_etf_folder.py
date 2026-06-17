"""

Build local folder `指月自选etf` containing kline parquet files for a watchlist of ETFs.



Watchlist source:

  - 指月自选etf/codes.tsv (columns: code, name, symbol)



Data source:

  - use AkShare daily ETF history with Sina fallback (via examples/update_research_cache.py)



Cache layout:

  {czsc_research_cache}/A股场内基金/{symbol}.parquet

"""



from __future__ import annotations



import os

import shutil

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path



import pandas as pd

from loguru import logger





GROUP_ETF = "A股场内基金"

DEFAULT_CACHE_ROOT = os.environ.get("czsc_research_cache", r"D:\CZSC投研数据")

DEFAULT_SDT = "2010-01-01"





@dataclass(frozen=True)

class WatchItem:

    code6: str

    exch: str

    symbol: str

    name: str





def _symbol_from_code(code6: str) -> str:

    code6 = str(code6).strip()

    if len(code6) != 6 or not code6.isdigit():

        raise ValueError(f"invalid 6-digit code: {code6!r}")

    if code6.startswith("159"):

        return f"{code6}.SZ"

    if code6.startswith(("51", "56", "58")):

        return f"{code6}.SH"

    raise ValueError(f"cannot infer exchange for code: {code6}")





def load_watchlist_from_tsv(path: Path) -> list[WatchItem]:

    df = pd.read_csv(path, sep="\t", encoding="utf-8")

    required = {"code", "name", "symbol"}

    if not required.issubset(df.columns):

        raise ValueError(f"TSV missing required columns: {path} got={df.columns.tolist()}")



    items: list[WatchItem] = []

    for _, row in df.iterrows():

        code6 = str(row["code"]).strip().zfill(6)

        name = str(row["name"]).strip()

        sym_raw = str(row["symbol"]).strip()

        if "." in sym_raw:

            code_from_sym, exch = sym_raw.split(".", 1)

            symbol = f"{code_from_sym}.{exch.upper()}"

            code6 = code_from_sym

        else:

            symbol = _symbol_from_code(code6)

            code6, exch = symbol.split(".")

        items.append(WatchItem(code6=code6, exch=exch, symbol=symbol, name=name))



    dedup: list[WatchItem] = []

    seen: set[str] = set()

    for it in items:

        if it.symbol in seen:

            continue

        seen.add(it.symbol)

        dedup.append(it)

    return dedup





def remove_stale_parquets(output_dir: Path, keep_symbols: set[str]) -> list[str]:

    removed: list[str] = []

    for p in output_dir.glob("*.parquet"):

        if p.stem not in keep_symbols:

            p.unlink()

            removed.append(p.name)

            logger.info(f"removed stale parquet: {p.name}")

    return removed





def ensure_cache_parquet(symbol: str, cache_root: Path, sdt: str, edt: str) -> Path:

    from examples.update_research_cache import fetch_etf_daily_akshare



    group_path = cache_root / GROUP_ETF

    group_path.mkdir(parents=True, exist_ok=True)

    path = group_path / f"{symbol}.parquet"

    if path.exists():

        return path



    logger.info(f"fetch missing: {symbol} ({sdt} ~ {edt})")

    df = fetch_etf_daily_akshare(symbol=symbol, sdt=sdt, edt=edt)

    if df is None or df.empty:

        raise RuntimeError(f"empty data fetched for {symbol}")

    keep = ["symbol", "dt", "open", "close", "high", "low", "vol", "amount"]

    df[keep].to_parquet(path, index=False)

    return path





def max_dt_in_parquet(path: Path) -> str:

    df = pd.read_parquet(path, columns=["dt"])

    if df.empty:

        return ""

    return pd.to_datetime(df["dt"]).max().strftime("%Y-%m-%d %H:%M:%S")





def main() -> int:

    repo_root = Path(__file__).resolve().parent

    output_dir = repo_root / "指月自选etf"

    output_dir.mkdir(parents=True, exist_ok=True)



    codes_tsv = output_dir / "codes.tsv"

    items = load_watchlist_from_tsv(codes_tsv)

    logger.info(f"watchlist size={len(items)} source={codes_tsv}")



    keep_symbols = {it.symbol for it in items}

    removed = remove_stale_parquets(output_dir, keep_symbols)

    if removed:

        logger.info(f"removed {len(removed)} stale parquet(s)")



    cache_root = Path(DEFAULT_CACHE_ROOT)

    edt = datetime.now().strftime("%Y-%m-%d")



    rows = []

    copied = 0

    failed = 0



    for it in items:

        status = "ok"

        err = ""

        cache_path = ""

        out_path = ""

        latest = ""

        try:

            p = ensure_cache_parquet(it.symbol, cache_root=cache_root, sdt=DEFAULT_SDT, edt=edt)

            cache_path = str(p)

            latest = max_dt_in_parquet(p)

            dst = output_dir / p.name

            shutil.copy2(p, dst)

            out_path = str(dst)

            copied += 1

        except Exception as exc:

            status = "error"

            err = str(exc)

            failed += 1

            logger.error(f"{it.symbol}: {exc}")



        rows.append(

            {

                "symbol": it.symbol,

                "name": it.name,

                "latest_dt": latest,

                "cache_parquet": cache_path,

                "output_parquet": out_path,

                "status": status,

                "error": err,

            }

        )



    manifest = pd.DataFrame(rows)

    manifest_path = output_dir / "manifest.tsv"

    manifest.to_csv(manifest_path, sep="\t", index=False, encoding="utf-8")

    logger.info(f"output_dir={output_dir} copied={copied} failed={failed} manifest={manifest_path}")



    if failed:

        logger.warning("Some symbols failed; see manifest.tsv for details")

        return 2

    return 0





if __name__ == "__main__":

    raise SystemExit(main())


