import datetime as dt

import numpy as np
import pandas as pd
import pytest

from screener import indicators as ind
from screener import realtime as rt
from screener import signals as sg

PAYLOAD = {"msgArray": [
    {"c": "0050", "n": "元大台灣50", "ex": "tse", "d": "20261105", "t": "10:15:05", "z": "97.1500", "pz": "97.1",
     "o": "99.4000", "h": "99.7500", "l": "96.9000", "y": "99.8500", "v": "423468",
     "b": "97.1000_97.0500_", "a": "97.1500_97.2000_", "u": "109.8000", "w": "89.9000"},
    {"c": "6488", "n": "環球晶", "ex": "otc", "d": "20261105", "t": "10:15:05", "z": "-", "pz": "564.0000",
     "o": "601.0000", "h": "619.0000", "l": "558.0000", "y": "619.0000", "v": "18169", "b": "563_", "a": "565_"},
    {"c": "9999", "n": "沒成交", "ex": "tse", "d": "20261105", "t": "09:00:00", "z": "-", "pz": "-",
     "o": "-", "h": "-", "l": "-", "y": "50", "v": "0", "b": "49.9_", "a": "50.1_"},
]}


def test_parse_fields_and_fallbacks():
    q = rt.parse(PAYLOAD)
    assert list(q.index) == ["0050", "6488", "9999"]
    a = q.loc["0050"]
    assert a["price"] == 97.15 and a["market"] == "TWSE" and a["volume"] == 423468 * 1000
    assert a["change_pct"] == pytest.approx(97.15 / 99.85 - 1) and a["date"] == pd.Timestamp("2026-11-05")
    assert q.loc["6488", "price"] == 564.0 and q.loc["6488", "market"] == "TPEx"   # z 是 "-" → 用 pz
    assert q.loc["9999", "price"] == pytest.approx(50.0)                           # 沒成交 → 買賣中間價
    assert rt.parse({}).empty


def _hist(closes, start="2026-01-02"):
    c = np.asarray(closes, float)
    return pd.DataFrame({"date": pd.bdate_range(start, periods=len(c)), "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1e6})


def _quote(price, date, prev=None):
    return pd.Series({"date": pd.Timestamp(date), "price": price, "open": price, "high": price, "low": price,
                      "volume": 5e5, "prev_close": prev or price})


def test_provisional_daily_appends_once():
    h = _hist([100.0] * 10)
    nxt = h["date"].iat[-1] + pd.offsets.BDay()
    full = rt.provisional_daily(h, _quote(95.0, nxt))
    assert len(full) == 11 and full["close"].iat[-1] == 95.0
    assert len(rt.provisional_daily(h, _quote(95.0, h["date"].iat[-1]))) == 10   # 今天已收盤 → 不重複接


def test_rsi_cross_price_hits_target():
    rng = np.random.default_rng(0)
    closes = pd.Series(100 + np.cumsum(rng.normal(0, 1, 80)))
    for target in (25, 70):
        p = rt.rsi_cross_price(closes, target, 3)
        assert p is not None
        got = ind.rsi(pd.concat([closes, pd.Series([p])], ignore_index=True), 3).iat[-1]
        assert got == pytest.approx(target, abs=0.01)


def test_intraday_check_flags_buy_when_close_drops():
    closes = list(np.linspace(80, 120, 90))                     # 多頭：收盤在季線之上
    h = _hist(closes)
    nxt = h["date"].iat[-1] + pd.offsets.BDay()
    calm = rt.intraday_check(h, _quote(120.5, nxt), tax=sg.TAX_ETF)
    assert calm["status"] == "flat" and calm["action"] is None and calm["buy_below"] < 120
    drop = rt.intraday_check(h, _quote(calm["buy_below"] * 0.995, nxt), tax=sg.TAX_ETF)
    assert drop["action"] == "buy" and drop["lines"][0].startswith("▲")


def test_market_open():
    tz = dt.timezone(dt.timedelta(hours=8))
    assert rt.market_open(dt.datetime(2026, 11, 5, 10, 0, tzinfo=tz))
    assert not rt.market_open(dt.datetime(2026, 11, 5, 14, 0, tzinfo=tz))
    assert not rt.market_open(dt.datetime(2026, 11, 7, 10, 0, tzinfo=tz))   # 星期六
