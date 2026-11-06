"""上市（TWSE）資料來源。

- 每日收盤行情：MI_INDEX，可指定日期，type=ALLBUT0999 = 全部（不含權證、牛熊證）
- 三大法人買賣超：T86，可指定日期
- 公司基本資料（產業別）：OpenAPI t187ap03_L，只有最新資料
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import http
from ..parse import SchemaError, institutional_table, margin_table, pick_table, price_table, to_number

BASE = "https://www.twse.com.tw/rwd/zh"
OPENAPI = "https://openapi.twse.com.tw/v1"


def _no_data(payload) -> bool:
    # On holidays TWSE answers {"stat": "很抱歉，沒有符合條件的資料!"}
    return not payload or str(payload.get("stat", "")).upper() != "OK"


def prices(day: dt.date) -> pd.DataFrame | None:
    payload = http.get_json(f"{BASE}/afterTrading/MI_INDEX", {
        "date": day.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"})
    if _no_data(payload):
        return None
    table = pick_table(payload, ["代號", "收盤"])
    if table is None:
        raise SchemaError(f"上市行情回應中找不到個股收盤表（{day}）")
    return price_table(*table, market="TWSE")


def institutional(day: dt.date) -> pd.DataFrame | None:
    payload = http.get_json(f"{BASE}/fund/T86", {
        "date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"})
    if _no_data(payload):
        return None
    table = pick_table(payload, ["代號", "三大法人"])
    if table is None:
        raise SchemaError(f"上市三大法人回應中找不到資料表（{day}）")
    return institutional_table(*table)


def margin(day: dt.date) -> pd.DataFrame | None:
    """融資融券彙總（MI_MARGN，selectType=ALL），單位張。"""
    payload = http.get_json(f"{BASE}/marginTrading/MI_MARGN", {
        "date": day.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
    if _no_data(payload):
        return None
    table = pick_table(payload, ["代號", "今日餘額"])
    if table is None:
        return None
    return margin_table(*table)


def stock_info() -> pd.DataFrame:
    rows = http.get_json(f"{OPENAPI}/opendata/t187ap03_L") or []
    return _info_frame(rows, "TWSE")


def _info_frame(rows: list[dict], market: str) -> pd.DataFrame:
    def pick(row: dict, *keys: str) -> str:
        for k in keys:
            if k in row and str(row[k]).strip():
                return str(row[k]).strip()
        return ""

    def shares(row: dict) -> float | None:
        n = to_number(pick(row, "已發行普通股數或TDR原股發行股數", "IssueShares"))
        if n:
            return n
        capital = to_number(pick(row, "實收資本額", "Paidin.Capital.NTDollars", "PaidInCapital"))
        return capital / 10 if capital else None  # 面額 10 元

    out = [{
        "code": pick(r, "公司代號", "SecuritiesCompanyCode", "Code"),
        "name": pick(r, "公司簡稱", "CompanyAbbreviation", "公司名稱", "CompanyName"),
        "market": market,
        "industry_code": pick(r, "產業別", "SecuritiesIndustryCode", "IndustryCode"),
        "shares": shares(r),
    } for r in rows]
    return pd.DataFrame(out, columns=["code", "name", "market", "industry_code", "shares"])
