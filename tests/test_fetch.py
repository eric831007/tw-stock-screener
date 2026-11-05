"""Offline tests: sample responses shaped like the real TWSE / TPEx answers."""
import datetime as dt

import pandas as pd
import pytest

from screener import fetch, http, parse, store
from screener.industry import industry_name
from screener.sources import tpex, twse

TWSE_PRICES = {
    "stat": "OK",
    "tables": [
        {"title": "價格指數", "fields": ["指數", "收盤指數"], "data": [["發行量加權股價指數", "23,000.12"]]},
        {"title": "每日收盤行情(全部(不含權證、牛熊證))",
         "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價",
                    "最低價", "收盤價", "漲跌(+/-)", "漲跌價差", "最後揭示買價", "本益比"],
         "data": [
             ["2330", "台積電", "25,123,456", "40,000", "25,000,000,000", "1,000.00", "1,010.00",
              "995.00", "1,005.00", "<p style= color:red>+</p>", "5.00", "1,005.00", "25.1"],
             ["0050", "元大台灣50", "10,000,000", "8,000", "2,000,000,000", "200.00", "201.00",
              "199.50", "200.50", "<p>+</p>", "0.50", "200.45", "0.00"],
             ["9999", "停牌股", "0", "0", "0", "--", "--", "--", "--", "", "0.00", "--", "0.00"],
             ["030001", "某權證", "1,000", "1", "1,000", "1.00", "1.00", "1.00", "1.00", "", "0", "1", "0"],
         ]},
    ],
}

TWSE_T86 = {
    "stat": "OK",
    "fields": ["證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)",
               "外陸資買賣超股數(不含外資自營商)", "外資自營商買進股數", "外資自營商賣出股數",
               "外資自營商買賣超股數", "投信買進股數", "投信賣出股數", "投信買賣超股數", "自營商買賣超股數",
               "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
               "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)", "三大法人買賣超股數"],
    "data": [["2330", "台積電", "9,000", "4,000", "5,000", "0", "0", "100", "800", "300", "500",
              "-200", "0", "0", "-150", "0", "0", "-50", "5,300"]],
}

TPEX_PRICES = {
    "stat": "ok",
    "tables": [{
        "fields": ["代號", "名稱", "收盤 ", "漲跌", "開盤 ", "最高 ", "最低", "均價 ", "成交股數  ",
                   "成交金額(元)", "成交筆數 "],
        "data": [["6488", "環球晶", "450.00", "+5.00", "445.00", "455.00", "444.00", "449.00",
                  "3,000,000", "1,347,000,000", "2,500"]],
    }],
}

TPEX_INST = {
    "tables": [{
        "fields": ["代號", "名稱",
                   "外資及陸資(不含外資自營商)-買進股數", "外資及陸資(不含外資自營商)-賣出股數",
                   "外資及陸資(不含外資自營商)-買賣超股數", "外資自營商-買賣超股數",
                   "外資及陸資-買賣超股數", "投信-買進股數", "投信-賣出股數", "投信-買賣超股數",
                   "自營商(自行買賣)-買賣超股數", "自營商(避險)-買賣超股數", "自營商-買賣超股數",
                   "三大法人買賣超股數合計"],
        "data": [["6488", "環球晶", "200,000", "50,000", "150,000", "0", "150,000", "30,000", "0",
                  "30,000", "1,000", "-3,000", "-2,000", "178,000"]],
    }],
}

HOLIDAY = {"stat": "很抱歉，沒有符合條件的資料!"}


def fake_get_json(responses):
    def _get(url, params=None):
        for key, value in responses.items():
            if key in url:
                return value
        raise AssertionError(f"unexpected url {url}")
    return _get


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    return tmp_path


def test_twse_prices_parse(monkeypatch):
    monkeypatch.setattr(http, "get_json", fake_get_json({"MI_INDEX": TWSE_PRICES}))
    df = twse.prices(dt.date(2026, 11, 4))
    assert list(df["code"]) == ["2330", "0050", "9999"]  # warrant dropped
    tsmc = df.iloc[0]
    assert tsmc["close"] == 1005.0 and tsmc["volume"] == 25_123_456 and tsmc["market"] == "TWSE"
    assert pd.isna(df.iloc[2]["close"])  # "--" becomes empty


def test_twse_institutional_picks_total_columns(monkeypatch):
    monkeypatch.setattr(http, "get_json", fake_get_json({"T86": TWSE_T86}))
    row = twse.institutional(dt.date(2026, 11, 4)).iloc[0]
    assert (row["foreign_net"], row["trust_net"], row["dealer_net"], row["total_net"]) == \
        (5000, 500, -200, 5300)


def test_tpex_parse(monkeypatch):
    monkeypatch.setattr(http, "get_json",
                        fake_get_json({"dailyQuotes": TPEX_PRICES, "dailyTrade": TPEX_INST}))
    p = tpex.prices(dt.date(2026, 11, 4)).iloc[0]
    assert (p["code"], p["close"], p["volume"], p["value"]) == ("6488", 450.0, 3_000_000, 1_347_000_000)
    i = tpex.institutional(dt.date(2026, 11, 4)).iloc[0]
    assert (i["foreign_net"], i["trust_net"], i["dealer_net"], i["total_net"]) == \
        (150_000, 30_000, -2_000, 178_000)


TPEX_INST_TWO_LEVEL = {
    "tables": [{
        "fields": ["代號", "名稱"] + ["買進股數", "賣出股數", "買賣超股數"] * 7 + ["三大法人買賣超股數合計"],
        # 外資(不含) 外資自營 外資合計 投信 自營(自行) 自營(避險) 自營合計 三大法人
        "data": [["6488", "環球晶", "200,000", "50,000", "150,000", "1,000", "0", "1,000",
                  "201,000", "50,000", "151,000", "30,000", "0", "30,000", "1,000", "0", "1,000",
                  "0", "3,000", "-3,000", "1,000", "3,000", "-2,000", "179,000"]],
    }],
}


def test_tpex_two_level_headers(monkeypatch):
    monkeypatch.setattr(http, "get_json", fake_get_json({"dailyTrade": TPEX_INST_TWO_LEVEL}))
    i = tpex.institutional(dt.date(2026, 11, 4)).iloc[0]
    assert (i["foreign_net"], i["trust_net"], i["dealer_net"], i["total_net"]) == \
        (150_000, 30_000, -2_000, 179_000)


def test_tpex_wrong_column_order_is_caught(monkeypatch):
    bad = {"tables": [dict(TPEX_INST_TWO_LEVEL["tables"][0])]}
    bad["tables"][0]["data"] = [TPEX_INST_TWO_LEVEL["tables"][0]["data"][0][:-1] + ["999,999"]]
    monkeypatch.setattr(http, "get_json", fake_get_json({"dailyTrade": bad}))
    with pytest.raises(parse.SchemaError, match="順序"):
        tpex.institutional(dt.date(2026, 11, 4))


def test_schema_change_fails_loudly():
    with pytest.raises(parse.SchemaError, match="收盤"):
        parse.price_table(["證券代號", "證券名稱", "成交股數", "成交金額", "開盤價", "最高價", "最低價"],
                          [], "TWSE")


def test_fetch_day_saves_all_four_files(data_dir, monkeypatch):
    responses = {"MI_INDEX": TWSE_PRICES, "T86": TWSE_T86,
                 "dailyQuotes": TPEX_PRICES, "dailyTrade": TPEX_INST}
    monkeypatch.setattr(http, "get_json", fake_get_json(responses))
    day = dt.date(2026, 11, 4)
    assert fetch.fetch_day(day) == "complete"
    assert store.is_complete(day)
    assert store.trading_days() == [day]
    prices = store.load("prices")
    assert set(prices["code"]) == {"2330", "0050", "9999", "6488"}
    assert prices["code"].map(len).min() == 4  # codes keep leading zeros
    # second run downloads nothing
    monkeypatch.setattr(http, "get_json", fake_get_json({}))
    assert fetch.fetch_day(day) == "complete"


def test_past_day_without_data_is_holiday(data_dir, monkeypatch):
    monkeypatch.setattr(http, "get_json", fake_get_json({"MI_INDEX": HOLIDAY}))
    day = dt.date(2026, 10, 9)  # a past weekday
    assert fetch.fetch_day(day) == "holiday"
    assert day in store.holidays()
    assert fetch.fetch_day(day) == "holiday"  # remembered, no request


def test_weekend_skipped(data_dir):
    assert fetch.fetch_day(dt.date(2026, 11, 7)) == "weekend"


def test_backfill_counts_trading_days(data_dir, monkeypatch):
    responses = {"MI_INDEX": TWSE_PRICES, "T86": TWSE_T86,
                 "dailyQuotes": TPEX_PRICES, "dailyTrade": TPEX_INST}
    monkeypatch.setattr(http, "get_json", fake_get_json(responses))
    fetch.backfill(3, end=dt.date(2026, 11, 9))  # Monday
    assert store.trading_days() == [dt.date(2026, 11, 5), dt.date(2026, 11, 6), dt.date(2026, 11, 9)]


def test_industry_names():
    assert industry_name("24", "2330") == "半導體業"
    assert industry_name("", "00878") == "ETF"
    assert industry_name("99", "1234") == "產業99"


def test_old_numbered_tables():
    payload = {"stat": "OK", "fields1": ["指數", "收盤指數"], "data1": [["x", "1"]],
               "fields9": TWSE_PRICES["tables"][1]["fields"], "data9": TWSE_PRICES["tables"][1]["data"]}
    fields, rows = parse.pick_table(payload, ["代號", "收盤"])
    assert len(rows) == 4


def test_tpex_old_three_group_layout(monkeypatch):
    old = {"tables": [{
        "fields": ["代號", "名稱"] + ["買進股數", "賣出股數", "買賣超股數"] * 3 + ["三大法人買賣超股數"],
        "data": [["6488", "環球晶", "200,000", "50,000", "150,000", "30,000", "0", "30,000",
                  "1,000", "3,000", "-2,000", "178,000"]],
    }]}
    monkeypatch.setattr(http, "get_json", fake_get_json({"dailyTrade": old}))
    i = tpex.institutional(dt.date(2010, 3, 4)).iloc[0]
    assert (i["foreign_net"], i["trust_net"], i["dealer_net"], i["total_net"]) == \
        (150_000, 30_000, -2_000, 178_000)


def test_backfill_range(data_dir, monkeypatch):
    responses = {"MI_INDEX": TWSE_PRICES, "T86": TWSE_T86,
                 "dailyQuotes": TPEX_PRICES, "dailyTrade": TPEX_INST}
    monkeypatch.setattr(http, "get_json", fake_get_json(responses))
    fetch.backfill_range(dt.date(2010, 3, 1), dt.date(2010, 3, 7))  # Mon..Sun
    assert len(store.trading_days()) == 5


def test_one_failed_day_does_not_stop_backfill(data_dir, monkeypatch):
    responses = {"MI_INDEX": TWSE_PRICES, "T86": TWSE_T86, "dailyTrade": TPEX_INST}
    def flaky(url, params=None):
        if "dailyQuotes" in url:
            if params and params.get("date") == "2010/03/03":
                raise RuntimeError("連續 5 次請求失敗")
            return TPEX_PRICES
        return fake_get_json(responses)(url, params)
    monkeypatch.setattr(http, "get_json", flaky)
    monkeypatch.setattr(fetch, "failures", [])
    fetch.backfill_range(dt.date(2010, 3, 1), dt.date(2010, 3, 5))
    assert len(store.trading_days()) == 5
    assert not store.is_complete(dt.date(2010, 3, 3))
    assert store.is_complete(dt.date(2010, 3, 4))
    assert len(fetch.failures) == 1 and "2010-03-03 TPEx prices" in fetch.failures[0]
