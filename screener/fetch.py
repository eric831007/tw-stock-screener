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
        fetch_optional(day)
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
            if store.has(kind, day, market) or not store.available(kind, day, market):
                continue
            df = _try(getattr(source, kind), day, market, kind)
            if df is _FAILED:
                continue
            if df is None or df.empty:
                if day < today_taipei():
                    log.warning("%s %s%s：交易所沒有這天的資料", day, store.MARKET_NAMES[market], store.KIND_NAMES[kind])
                else:
                    log.warning("%s %s%s：還沒公布", day, store.MARKET_NAMES[market], store.KIND_NAMES[kind])
                continue
            _save(kind, day, market, df)

    complete = store.is_complete(day)
    if store.has("prices", day, "TWSE"):
        fetch_optional(day)
    return "complete" if complete else "partial"


def fetch_optional(day: dt.date) -> None:
    """融資融券等選用資料：還沒存的才抓；抓不到（還沒公布或失敗）只記 log，不算這天失敗。"""
    for kind in store.OPTIONAL_KINDS:
        for market, source in MARKETS.items():
            if store.has(kind, day, market) or not hasattr(source, kind) or not store.available(kind, day, market):
                continue
            try:
                df = getattr(source, kind)(day)
            except Exception as exc:
                log.warning("%s %s %s 抓取失敗（選用資料，下次再補）：%s", day, market, kind, exc)
                continue
            if df is None or df.empty:
                log.info("%s %s %s 尚未公布", day, market, kind)
                continue
            _save(kind, day, market, df)


def catch_up_optional(n_days: int = 5) -> None:
    """補最近 n 個已存交易日缺的選用資料（例如融資融券比排程晚公布）。"""
    for day in store.trading_days()[-n_days:]:
        fetch_optional(day)


def update_holders(min_gap_days: int = 8) -> str | None:
    """集保股權分散表（每週公布）。上次存的資料日期（週五）已經超過 min_gap_days 天才去下載，
    避免下一週資料還沒公布前每天重複下載（約 2.4 MB）。回傳存下的日期。"""
    from .sources import tdcc
    days = store.holders_days()
    if days and (today_taipei() - days[-1]).days < min_gap_days:
        return None
    try:
        got = tdcc.holders_latest()
    except Exception as exc:
        log.warning("集保股權分散表抓取失敗（選用資料）：%s", exc)
        return None
    if not got:
        return None
    day_s, df = got
    day = dt.date.fromisoformat(day_s)
    if not store.has(store.HOLDERS, day, "ALL"):
        _save(store.HOLDERS, day, "ALL", df)
    return day_s


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


def _note_unavailable(start: dt.date, end: dt.date) -> None:
    """回補期間有早於資料起始日的部分時，先講清楚哪些資料不會有。"""
    for (kind, market), first in sorted(store.EARLIEST.items(), key=lambda x: x[1]):
        if start < first and (kind in store.KINDS or kind in store.OPTIONAL_KINDS):
            log.info("說明：%s%s 從 %s 才有資料，%s 以前的日子不會抓這一項",
                     store.MARKET_NAMES[market], store.KIND_NAMES[kind], first, first)


def backfill_range(start: dt.date, end: dt.date) -> None:
    """Fetch every weekday from `end` back to `start` (inclusive)."""
    _note_unavailable(start, end)
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
