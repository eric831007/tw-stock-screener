"""集保結算所：股權分散表（每週，opendata 只提供最新一週）。"""
from __future__ import annotations

import pandas as pd

from .. import http
from ..parse import holders_table

URL = "https://opendata.tdcc.com.tw/getOD.ashx"


def holders_latest() -> tuple[str, pd.DataFrame] | None:
    """回傳（資料日期 YYYY-MM-DD, 每檔摘要），沒有資料時回傳 None。"""
    text = http.get_text(URL, {"id": "1-5"})
    if not text:
        return None
    return holders_table(text)
