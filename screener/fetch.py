"""Download a trading day (or a range of past days) and store it locally."""
from __future__ import annotations

import datetime as dt
import logging
from zoneinfo import ZoneInfo

import pandas as pd

from . import store
from .industry import industry_name
from .sources import MARKETS

log = logging.getLogger(__name__)
TAIPEI = ZoneInfo("Asia/Taipei")

# 這次執行中抓取失敗的項目，例如 "2026-06-17 TPEx prices：連續 5 次請求失敗"
failures: list[str] = []


def today_taipei() -> dt.date:
    return dt.datetime.now(TAIPEI).date()


def fetch_day(day: dt.date) -> str:
    """Fetch whatever is missing for `day`.

    Returns one of: "weekend", "holiday", "complete", "partial", "not-published".
    Raises on network failures or changed response formats, so the run fails
    loudly instead of quietly producing a wrong ranking later.
    """
    if day.weekday() >= 5:
        return "weekend"
    if day in store.holidays():
        return "holiday"
    if store.is_complete(day):
        return "complete"

    # TWSE prices first: no TWSE prices on a past weekday means the market was closed.
    if not store.has("prices", day, "TWSE"):
        df = _try(MARKETS["TWSE"].prices, day, "TWSE", "prices")
        if df is _FAILED:
            return "partial"
        if df is None:
            if day < today_taipei():
                store.mark_holiday(day)
                log.info("%s 休市", day)
                return "holiday"
            log.warning("%s 上市行情尚未公布", day)
            return "not-published"
        _save("prices", day, "TWSE", df)

    for kind in store.KINDS:
        for market, source in MARKETS.items():
            if store.has(kind, day, market):
                continue
            df = _try(getattr(source, kind), day, market, kind)
            if df is _FAILED:
                continue
            if df is None or df.empty:
                log.warning("%s %s %s 尚無資料（可能還沒公布）", day, market, kind)
                continue
            _save(kind, day, market, df)

    return "complete" if store.is_complete(day) else "partial"


_FAILED = object()


def _try(func, day: dt.date, market: str, kind: str):
    """Run one download; on failure log it, remember it, and let the caller move on."""
    try:
        return func(day)
    except Exception as exc:
        failures.append(f"{day} {market} {kind}：{exc}")
        log.error("%s %s %s 抓取失敗，先跳過，下次執行會再補：%s", day, market, kind, exc)
        return _FAILED


def _save(kind: str, day: dt.date, market: str, df: pd.DataFrame) -> None:
    path = store.save(kind, day, market, df)
    log.info("已存 %s（%d 筆）", path.relative_to(store.DATA_DIR.parent), len(df))


def backfill(trading_days: int, end: dt.date | None = None) -> None:
    """Walk back from `end` until `trading_days` trading days are stored."""
    day = end or today_taipei()
    found = 0
    limit = trading_days * 2 + 30  # calendar days; guards against endless loops
    for _ in range(limit):
        status = fetch_day(day)
        if status in {"complete", "partial"}:
            found += 1
            log.info("回補進度 %d/%d（%s %s）", found, trading_days, day, status)
            if found >= trading_days:
                return
        day -= dt.timedelta(days=1)
    log.warning("往回 %d 天只找到 %d 個交易日", limit, found)


def backfill_range(start: dt.date, end: dt.date) -> None:
    """Fetch every weekday from `end` back to `start` (inclusive)."""
    day, found = end, 0
    while day >= start:
        status = fetch_day(day)
        if status in {"complete", "partial"}:
            found += 1
            if found % 20 == 0:
                log.info("進度：%s，已完成 %d 個交易日", day, found)
        day -= dt.timedelta(days=1)
    log.info("%s ～ %s 共 %d 個交易日", start, end, found)


def update_stock_info() -> None:
    frames = []
    for market, source in MARKETS.items():
        try:
            frames.append(source.stock_info())
        except Exception as exc:  # industry is nice-to-have; never block the daily fetch
            log.warning("%s 公司基本資料抓取失敗：%s", market, exc)
    if not frames:
        return
    info = pd.concat(frames, ignore_index=True)
    info = info[info["code"].astype(str).str.len() > 0].drop_duplicates("code")
    info["industry"] = [industry_name(c, s) for c, s in zip(info["industry_code"], info["code"])]
    store.save_stock_info(info)
    log.info("已更新產業別（%d 檔）", len(info))
