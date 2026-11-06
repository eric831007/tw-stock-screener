"""Helpers for turning the exchanges' JSON tables into clean pandas DataFrames.

The exchanges return every value as a string ("1,234", "--", "X0.00") and
their column headers sometimes carry stray spaces or change slightly over
time, so columns are found by keyword rather than by exact name.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

# Keep ordinary stocks (4 digits) and ETFs (00 + 2-4 digits + optional letter,
# e.g. 0050, 00878, 00632R, 00679B). Drops warrants, CBs and other products.
CODE_PATTERN = re.compile(r"^(\d{4}|00\d{2,4}[A-Z]?)$")


class SchemaError(RuntimeError):
    """Raised when an exchange response no longer looks the way we expect."""


def to_number(value) -> float | None:
    """'1,234.5' -> 1234.5 ; '--', '', 'X0.00' -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"<[^>]+>", "", str(value))  # strip HTML such as <p>+</p>
    text = text.replace(",", "").strip()
    if text in {"", "--", "---", "----", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_headers(fields: Sequence[str]) -> list[str]:
    return [re.sub(r"\s+", "", str(f)) for f in fields]


def find_column(headers: Sequence[str], *, starts: str | None = None,
                contains: Iterable[str] = (), excludes: Iterable[str] = (),
                prefer: Iterable[str] = (), required: bool = True,
                what: str = "") -> int | None:
    """Return the index of the header matching the rules.

    starts:   header must start with this text
    contains: header must contain every one of these
    excludes: header must contain none of these
    prefer:   among several matches, pick the first containing one of these;
              otherwise the shortest header (usually the "total" column)
    """
    contains, excludes, prefer = list(contains), list(excludes), list(prefer)
    matches = []
    for i, h in enumerate(headers):
        if starts and not h.startswith(starts):
            continue
        if any(c not in h for c in contains):
            continue
        if any(e in h for e in excludes):
            continue
        matches.append(i)
    if not matches:
        if required:
            raise SchemaError(
                f"找不到欄位「{what or starts or contains}」，交易所回傳的欄位是：{list(headers)}"
            )
        return None
    for p in prefer:
        for i in matches:
            if p in headers[i]:
                return i
    return min(matches, key=lambda i: len(headers[i]))


def filter_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["code"] = df["code"].astype(str).str.strip()
    return df[df["code"].str.match(CODE_PATTERN)].reset_index(drop=True)


def price_table(fields: Sequence[str], rows: Sequence[Sequence], market: str) -> pd.DataFrame:
    """Build the standard daily price frame from a raw exchange table."""
    h = clean_headers(fields)
    col = {
        "code": find_column(h, contains=["代號"], what="代號"),
        "name": find_column(h, contains=["名稱"], what="名稱"),
        "volume": find_column(h, starts="成交股數", what="成交股數"),
        "value": find_column(h, starts="成交金額", what="成交金額"),
        "open": find_column(h, starts="開盤", what="開盤"),
        "high": find_column(h, starts="最高", what="最高"),
        "low": find_column(h, starts="最低", what="最低"),
        "close": find_column(h, starts="收盤", what="收盤"),
    }
    records = []
    for r in rows:
        records.append({
            "code": str(r[col["code"]]).strip(),
            "name": str(r[col["name"]]).strip(),
            "market": market,
            "open": to_number(r[col["open"]]),
            "high": to_number(r[col["high"]]),
            "low": to_number(r[col["low"]]),
            "close": to_number(r[col["close"]]),
            "volume": to_number(r[col["volume"]]),
            "value": to_number(r[col["value"]]),
        })
    df = pd.DataFrame.from_records(records, columns=[
        "code", "name", "market", "open", "high", "low", "close", "volume", "value"])
    return filter_codes(df)


def institutional_table(fields: Sequence[str], rows: Sequence[Sequence]) -> pd.DataFrame:
    """Build the standard three-institution net-buy frame (units: shares)."""
    # 櫃買中心舊版（約 2013～2018 年初）把「買賣超」寫成「淨買」，自營商合計寫成「自營淨買股數」
    h = [x.replace("淨買", "買賣超") for x in clean_headers(fields)]
    h = [("自營商" + x[2:]) if x.startswith("自營買賣超") else x for x in h]
    col = {
        "code": find_column(h, contains=["代號"], what="代號"),
        # Foreign & mainland investors, excluding foreign dealers.
        "foreign_net": find_column(h, contains=["外", "買賣超"], excludes=["三大法人"],
                                   prefer=["不含外資自營商", "不含"], what="外資買賣超"),
        "trust_net": find_column(h, contains=["投信", "買賣超"], what="投信買賣超"),
        # Dealers total (not the 自行買賣 / 避險 sub-columns, not foreign dealers).
        "dealer_net": find_column(h, contains=["自營商", "買賣超"],
                                  excludes=["外資", "外陸資", "自行", "避險"], what="自營商買賣超"),
        "total_net": find_column(h, contains=["三大法人", "買賣超"], what="三大法人買賣超"),
    }
    records = []
    for r in rows:
        records.append({
            "code": str(r[col["code"]]).strip(),
            "foreign_net": to_number(r[col["foreign_net"]]),
            "trust_net": to_number(r[col["trust_net"]]),
            "dealer_net": to_number(r[col["dealer_net"]]),
            "total_net": to_number(r[col["total_net"]]),
        })
    df = pd.DataFrame.from_records(records, columns=[
        "code", "foreign_net", "trust_net", "dealer_net", "total_net"])
    return filter_codes(df)


def pick_table(payload: dict, must_have: Sequence[str]) -> tuple[list, list] | None:
    """Find the (fields, data) table whose headers contain all of must_have.

    Handles both shapes the exchanges use:
      {"fields": [...], "data": [...]}            (single table)
      {"fields9": [...], "data9": [...]}          (older numbered tables)
      {"tables": [{"fields": [...], "data": [...]}, ...]}  (new style)
    """
    candidates = []
    if "fields" in payload and "data" in payload:
        candidates.append(payload)
    candidates.extend(t for t in payload.get("tables", []) if isinstance(t, dict))
    # Older TWSE responses use numbered keys: fields1/data1 ... fields9/data9
    for key in sorted(payload):
        m = re.fullmatch(r"fields(\d+)", key)
        if m and f"data{m.group(1)}" in payload:
            candidates.append({"fields": payload[key], "data": payload[f"data{m.group(1)}"]})
    for t in candidates:
        h = clean_headers(t.get("fields") or [])
        if all(any(m in x for x in h) for m in must_have) and t.get("data"):
            return t["fields"], t["data"]
    return None


MARGIN_COLUMNS = ["code", "margin_buy", "margin_sell", "margin_bal", "short_sell", "short_buy", "short_bal", "offset"]


def margin_table(fields: Sequence[str], rows: Sequence[Sequence]) -> pd.DataFrame:
    """融資融券（單位：張）。

    上市（MI_MARGN）的欄名重複：前一組「買進／賣出／今日餘額」是融資、後一組是融券；
    上櫃（margin/balance）的欄名不重複：資買、資賣、資餘額、券賣、券買、券餘額。
    """
    h = clean_headers(fields)
    if "資餘額" in h:   # 上櫃
        col = {"code": find_column(h, contains=["代號"], what="代號"),
               "margin_buy": h.index("資買"), "margin_sell": h.index("資賣"), "margin_bal": h.index("資餘額"),
               "short_sell": h.index("券賣"), "short_buy": h.index("券買"), "short_bal": h.index("券餘額"),
               "offset": find_column(h, contains=["資券相抵"], what="資券相抵")}
    else:              # 上市：同名欄位依出現順序分成融資、融券
        def nth(name: str, n: int) -> int:
            idx = [i for i, x in enumerate(h) if x == name]
            if len(idx) <= n:
                raise SchemaError(f"融資融券表找不到第 {n + 1} 個「{name}」欄位：{h}")
            return idx[n]
        col = {"code": find_column(h, contains=["代號"], what="代號"),
               "margin_buy": nth("買進", 0), "margin_sell": nth("賣出", 0), "margin_bal": nth("今日餘額", 0),
               "short_buy": nth("買進", 1), "short_sell": nth("賣出", 1), "short_bal": nth("今日餘額", 1),
               "offset": find_column(h, contains=["資券互抵"], what="資券互抵")}
    records = [{k: (str(r[i]).strip() if k == "code" else to_number(r[i])) for k, i in col.items()} for r in rows]
    return filter_codes(pd.DataFrame.from_records(records, columns=MARGIN_COLUMNS))


HOLDER_COLUMNS = ["code", "holders", "holders_1000", "pct_1000", "pct_400", "pct_le50"]


def holders_table(text: str) -> tuple[str, pd.DataFrame]:
    """集保股權分散表（opendata 1-5，CSV）→（資料日期 YYYY-MM-DD, 每檔摘要）。

    持股分級：1～8 級＝50 張以下（散戶），12～15 級＝400 張以上（大戶），15 級＝1,000 張以上（千張大戶），
    17 級＝合計。pct_* 是占集保庫存的百分比。
    """
    import io
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    raw.columns = clean_headers(raw.columns)
    date_col = find_column(list(raw.columns), contains=["日期"], what="資料日期")
    code_col = find_column(list(raw.columns), contains=["代號"], what="證券代號")
    lvl_col = find_column(list(raw.columns), contains=["分級"], what="持股分級")
    ppl_col = find_column(list(raw.columns), contains=["人數"], what="人數")
    pct_col = find_column(list(raw.columns), contains=["比例"], what="占集保庫存數比例")
    df = pd.DataFrame({"code": raw.iloc[:, code_col].str.strip(), "lvl": pd.to_numeric(raw.iloc[:, lvl_col]),
                       "people": pd.to_numeric(raw.iloc[:, ppl_col]), "pct": pd.to_numeric(raw.iloc[:, pct_col])})
    day = str(raw.iloc[0, date_col]).strip()
    day = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    piv = df.pivot_table(index="code", columns="lvl", values=["people", "pct"], aggfunc="sum")
    pct = piv["pct"].reindex(columns=range(1, 18), fill_value=0)
    ppl = piv["people"].reindex(columns=range(1, 18), fill_value=0)
    out = pd.DataFrame({"code": pct.index, "holders": ppl[17].to_numpy(), "holders_1000": ppl[15].to_numpy(),
                        "pct_1000": pct[15].to_numpy(), "pct_400": pct[[12, 13, 14, 15]].sum(axis=1).to_numpy(),
                        "pct_le50": pct[list(range(1, 9))].sum(axis=1).to_numpy()})
    out = filter_codes(out[HOLDER_COLUMNS])
    return day, out[~out["code"].str.startswith("000")].reset_index(drop=True)   # 000 開頭不是股票／ETF
