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
    h = clean_headers(fields)
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
      {"fields": [...], "data": [...]}            (old style)
      {"tables": [{"fields": [...], "data": [...]}, ...]}  (new style)
    """
    candidates = []
    if "fields" in payload and "data" in payload:
        candidates.append(payload)
    candidates.extend(t for t in payload.get("tables", []) if isinstance(t, dict))
    for t in candidates:
        h = clean_headers(t.get("fields") or [])
        if all(any(m in x for x in h) for m in must_have) and t.get("data"):
            return t["fields"], t["data"]
    return None
