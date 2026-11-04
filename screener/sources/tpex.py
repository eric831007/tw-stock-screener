"""上櫃（TPEx）資料來源。

- 每日收盤行情：afterTrading/dailyQuotes，可指定日期
- 三大法人買賣明細：insti/dailyTrade（sect=EW 全部、type=Daily 日報），可指定日期
- 公司基本資料（產業別）：OpenAPI mopsfin_t187ap03_O，只有最新資料
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import http
from ..parse import SchemaError, clean_headers, institutional_table, pick_table, price_table, to_number
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
    fields, rows = table
    return institutional_table(_expand_headers(fields, rows), rows)


# 櫃買中心的三大法人表是兩層表頭：第二層每欄只寫「買進股數／賣出股數／買賣超股數」，
# 類別放在第一層。依櫃買中心固定的欄位順序，把類別名稱補回每一欄前面。
INST_GROUPS = ["外資及陸資(不含外資自營商)", "外資自營商", "外資及陸資", "投信",
               "自營商(自行買賣)", "自營商(避險)", "自營商"]


def _expand_headers(fields, rows) -> list[str]:
    h = clean_headers(fields)
    if len(set(h)) == len(h):
        return h  # headers already unique: nothing to do
    expected = 2 + 3 * len(INST_GROUPS) + 1
    if len(h) != expected or not h[-1].startswith("三大法人"):
        raise SchemaError(f"上櫃三大法人欄位數與預期不同（{len(h)} 欄，預期 {expected}）：{h}")
    out = h[:2]
    for i, group in enumerate(INST_GROUPS):
        out += [f"{group}-{x}" for x in h[2 + 3 * i: 5 + 3 * i]]
    out.append(h[-1])
    _check_order(out, rows)
    return out


def _check_order(headers: list[str], rows) -> None:
    """外資及陸資(含外資自營商) + 投信 + 自營商 必須等於三大法人合計，否則欄位順序猜錯了。"""
    idx = {name: headers.index(name) for name in
           ("外資及陸資-買賣超股數", "投信-買賣超股數", "自營商-買賣超股數")}
    total = len(headers) - 1
    checked = bad = 0
    for r in rows:
        values = [to_number(r[i]) for i in (*idx.values(), total)]
        if None in values:
            continue
        checked += 1
        if abs(sum(values[:3]) - values[3]) > 1:
            bad += 1
    if checked and bad / checked > 0.01:
        raise SchemaError(f"上櫃三大法人欄位順序與預期不同：{bad}/{checked} 檔加總對不上合計")


def stock_info() -> pd.DataFrame:
    rows = http.get_json(f"{OPENAPI}/mopsfin_t187ap03_O") or []
    return _info_frame(rows, "TPEx")
