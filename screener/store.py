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
KINDS = ("prices", "institutional")
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


def is_complete(day: dt.date) -> bool:
    return all(has(k, day, m) for k in KINDS for m in MARKETS)


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


def load(kind: str, last_n_days: int | None = None) -> pd.DataFrame:
    """All stored rows of one kind with a `date` column, oldest first."""
    days = trading_days()
    if last_n_days:
        days = days[-last_n_days:]
    frames = []
    for day in days:
        for m in MARKETS:
            p = _path(kind, day, m)
            if p.exists():
                df = pd.read_csv(p, dtype={"code": str}, encoding=ENCODING)
                df.insert(0, "date", pd.Timestamp(day))
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_stock_info(df: pd.DataFrame) -> None:
    (DATA_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "stock_info.csv", index=False, encoding=ENCODING)


def load_stock_info() -> pd.DataFrame:
    f = DATA_DIR / "stock_info.csv"
    if not f.exists():
        return pd.DataFrame(columns=["code", "name", "market", "industry"])
    return pd.read_csv(f, dtype={"code": str, "industry_code": str}, encoding=ENCODING)
