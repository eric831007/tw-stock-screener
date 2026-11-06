"""Local history kept as small CSV files, one per market per trading day.

    data/prices/2026/2026-11-04_TWSE.csv
    data/institutional/2026/2026-11-04_TPEx.csv
    data/holidays.csv            dates confirmed as non-trading days
    data/stock_info.csv          code, name, market, industry

Plain CSV keeps the git history small and readable when GitHub Actions
commits the new day's files, and the files open directly in Excel.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("SCREENER_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
# 額外的唯讀歷史資料夾（例如從 Releases 解壓的 2023–2025 年），用作業系統的路徑分隔符號（Windows 是 ;）隔開。
# load() 會一起讀；同一天兩邊都有時以 DATA_DIR 為準。抓資料、判斷「已存過」只看 DATA_DIR。
EXTRA_DIRS = [Path(p) for p in os.environ.get("SCREENER_EXTRA_DATA_DIRS", "").split(os.pathsep) if p]
KINDS = ("prices", "institutional")       # 必要資料：缺了就算這天沒抓完
OPTIONAL_KINDS = ("margin",)              # 選用資料：抓不到不影響「這天是否完整」（融資融券）
HOLDERS = "holders"                       # 集保股權分散表：每週一個檔，市場欄位用 ALL
MARKETS = ("TWSE", "TPEx")
ENCODING = "utf-8-sig"  # BOM so Excel shows Chinese names correctly


def _path(kind: str, day: dt.date, market: str) -> Path:
    return DATA_DIR / kind / f"{day.year}" / f"{day.isoformat()}_{market}.csv"


def has(kind: str, day: dt.date, market: str) -> bool:
    return _path(kind, day, market).exists()


def save(kind: str, day: dt.date, market: str, df: pd.DataFrame) -> Path:
    path = _path(kind, day, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=ENCODING)
    return path


# 各資料來源最早有資料的日期（2026-11-06 實測）。在這之前交易所沒有提供，不用去抓，也不算缺資料。
# 上市股價約從 2004 年初開始；更早的日子抓不到上市股價會被當成休市。
EARLIEST = {
    ("prices", "TPEx"): dt.date(2007, 4, 23),
    ("institutional", "TWSE"): dt.date(2012, 5, 2),
    ("institutional", "TPEx"): dt.date(2014, 12, 1),
    ("margin", "TWSE"): dt.date(2001, 1, 2),
    ("margin", "TPEx"): dt.date(2007, 1, 2),
}
KIND_NAMES = {"prices": "股價", "institutional": "三大法人", "margin": "融資融券"}
MARKET_NAMES = {"TWSE": "上市", "TPEx": "上櫃"}


def available(kind: str, day: dt.date, market: str) -> bool:
    """交易所在這一天有沒有提供這種資料。"""
    start = EARLIEST.get((kind, market))
    return start is None or day >= start


def is_complete(day: dt.date) -> bool:
    """必要資料都存了（交易所那天還沒開始提供的不算缺）。"""
    return all(has(k, day, m) or not available(k, day, m) for k in KINDS for m in MARKETS)


# ---- holidays -------------------------------------------------------------

def _holiday_file() -> Path:
    return DATA_DIR / "holidays.csv"


def holidays() -> set[dt.date]:
    f = _holiday_file()
    if not f.exists():
        return set()
    return {dt.date.fromisoformat(d) for d in pd.read_csv(f)["date"].astype(str)}


def mark_holiday(day: dt.date) -> None:
    days = holidays() | {day}
    f = _holiday_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": sorted(d.isoformat() for d in days)}).to_csv(f, index=False)


# ---- reading history ------------------------------------------------------

def trading_days() -> list[dt.date]:
    """Dates that have a TWSE price file, oldest first."""
    folder = DATA_DIR / "prices"
    if not folder.exists():
        return []
    days = {dt.date.fromisoformat(p.name.split("_")[0]) for p in folder.glob("*/*_TWSE.csv")}
    return sorted(days)


def _day_files(kind: str) -> dict[dt.date, dict[str, Path]]:
    """{日期: {市場: 檔案}}，DATA_DIR 優先，其次 EXTRA_DIRS。"""
    out: dict[dt.date, dict[str, Path]] = {}
    for root in [DATA_DIR, *EXTRA_DIRS]:
        folder = root / kind
        if not folder.exists():
            continue
        for p in folder.glob("*/*.csv"):
            day_s, _, market = p.stem.partition("_")
            try:
                day = dt.date.fromisoformat(day_s)
            except ValueError:
                continue
            out.setdefault(day, {}).setdefault(market, p)
    return out


def load(kind: str, last_n_days: int | None = None) -> pd.DataFrame:
    """All stored rows of one kind with a `date` column, oldest first（含 EXTRA_DIRS 的歷史）。"""
    files = _day_files(kind)
    days = sorted(d for d, ms in files.items() if "TWSE" in ms) if kind == "prices" else sorted(files)
    if last_n_days:
        days = days[-last_n_days:]
    frames = []
    for day in days:
        for m, p in sorted(files.get(day, {}).items()):
            if p.exists():
                df = pd.read_csv(p, dtype={"code": str}, encoding=ENCODING)
                df.insert(0, "date", pd.Timestamp(day))
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def holders_days() -> list[dt.date]:
    """已存的集保資料日期（只看 DATA_DIR）。"""
    return sorted(d for d, ms in _day_files(HOLDERS).items() if any((DATA_DIR / HOLDERS) in p.parents for p in ms.values()))


def save_stock_info(df: pd.DataFrame) -> None:
    (DATA_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "stock_info.csv", index=False, encoding=ENCODING)


def load_stock_info() -> pd.DataFrame:
    f = DATA_DIR / "stock_info.csv"
    if not f.exists():
        return pd.DataFrame(columns=["code", "name", "market", "industry"])
    return pd.read_csv(f, dtype={"code": str, "industry_code": str}, encoding=ENCODING)
