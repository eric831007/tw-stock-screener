"""上櫃（TPEx）資料來源。

- 每日收盤行情：afterTrading/dailyQuotes，可指定日期
- 三大法人買賣明細：insti/dailyTrade（sect=EW 全部、type=Daily 日報），可指定日期
- 公司基本資料（產業別）：OpenAPI mopsfin_t187ap03_O，只有最新資料
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import http
from ..parse import institutional_table, pick_table, price_table
from .twse import _info_frame

BASE = "https://www.tpex.org.tw/www/zh-tw"
OPENAPI = "https://www.tpex.org.tw/openapi/v1"


def _date(day: dt.date) -> str:
    return day.strftime("%Y/%m/%d")


def prices(day: dt.date) -> pd.DataFrame | None:
    payload = http.get_json(f"{BASE}/afterTrading/dailyQuotes",
                            {"date": _date(day), "id": "", "response": "json"})
    table = pick_table(payload or {}, ["代號", "收盤"])
    if table is None:
        return None  # holiday or not yet published
    return price_table(*table, market="TPEx")


def institutional(day: dt.date) -> pd.DataFrame | None:
    payload = http.get_json(f"{BASE}/insti/dailyTrade",
                            {"type": "Daily", "sect": "EW", "date": _date(day), "response": "json"})
    table = pick_table(payload or {}, ["代號", "三大法人"])
    if table is None:
        return None
    return institutional_table(*table)


def stock_info() -> pd.DataFrame:
    rows = http.get_json(f"{OPENAPI}/mopsfin_t187ap03_O") or []
    return _info_frame(rows, "TPEx")
