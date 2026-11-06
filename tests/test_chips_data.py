"""融資融券、集保股權分散表的解析（欄位格式取自 2026-11-05 實際回應）。"""
import datetime as dt

import pandas as pd
import pytest

from screener import fetch, parse, store

TWSE_FIELDS = ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額", "次一營業日限額",
               "買進", "賣出", "現券償還", "前日餘額", "今日餘額", "次一營業日限額", "資券互抵", "註記"]
TWSE_ROWS = [["2330", "台積電", "1,738", "1,093", "71", "16,207", "16,781", "581,035",
              "12", "34", "0", "500", "522", "581,035", "5", " "],
             ["0050", "元大台灣50", "100", "50", "0", "1,000", "1,050", "9", "0", "0", "0", "0", "0", "9", "0", ""]]
TPEX_FIELDS = ["代號", "名稱", "前資餘額(張)", "資買", "資賣", "現償", "資餘額", "資屬證金", "資使用率(%)", "資限額",
               "前券餘額(張)", "券賣", "券買", "券償", "券餘額", "券屬證金", "券使用率(%)", "券限額", "資券相抵(張)", "備註"]
TPEX_ROWS = [["6488", "環球晶", "16,881", "1,074", "742", "0", "17,213", "79", "14.66", "117,394",
              "105", "7", "2", "0", "110", "0", "0.08", "117,394", "3", ""]]


def test_margin_table_twse_duplicate_headers():
    df = parse.margin_table(TWSE_FIELDS, TWSE_ROWS).set_index("code")
    r = df.loc["2330"]
    assert (r.margin_buy, r.margin_sell, r.margin_bal) == (1738, 1093, 16781)
    assert (r.short_buy, r.short_sell, r.short_bal, r.offset) == (12, 34, 522, 5)
    assert list(df.index) == ["2330", "0050"]


def test_margin_table_tpex():
    r = parse.margin_table(TPEX_FIELDS, TPEX_ROWS).iloc[0]
    assert r.code == "6488" and (r.margin_buy, r.margin_sell, r.margin_bal) == (1074, 742, 17213)
    assert (r.short_sell, r.short_buy, r.short_bal, r.offset) == (7, 2, 110, 3)


def test_margin_table_schema_change_is_loud():
    with pytest.raises(parse.SchemaError):
        parse.margin_table(["代號", "名稱", "買進", "今日餘額"], [["2330", "x", "1", "2"]])


def _tdcc_csv():
    lines = ["資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%"]
    pcts = [1.18, 3.46, 1.50, 0.84, 0.56, 0.75, 0.49, 0.35, 1.11, 1.12, 1.42, 1.09, 0.91, 0.75, 84.38, 0.0, 100.0]
    ppl = [2610784, 469863, 54468, 17931, 8385, 8026, 3719, 2049, 4156, 2087, 1321, 581, 342, 219, 1478, 1, 3185409]
    for lvl, (n, p) in enumerate(zip(ppl, pcts), start=1):
        lines.append(f"20261030,2330  ,{lvl},{n},0,{p}")
    lines.append("20261030,000218,17,0,0,100.00")   # 非股票代號，應被濾掉
    return "\n".join(lines)


def test_holders_table_summary():
    day, df = parse.holders_table(_tdcc_csv())
    assert day == "2026-10-30" and list(df["code"]) == ["2330"]
    r = df.iloc[0]
    assert r.holders == 3185409 and r.holders_1000 == 1478 and r.pct_1000 == pytest.approx(84.38)
    assert r.pct_400 == pytest.approx(1.09 + 0.91 + 0.75 + 84.38)
    assert r.pct_le50 == pytest.approx(sum([1.18, 3.46, 1.50, 0.84, 0.56, 0.75, 0.49, 0.35]))


class FakeSource:
    def __init__(self, df):
        self.df, self.calls = df, 0

    def margin(self, day):
        self.calls += 1
        if self.df is RuntimeError:
            raise RuntimeError("timeout")
        return self.df


def test_fetch_optional_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    ok = FakeSource(parse.margin_table(TWSE_FIELDS, TWSE_ROWS))
    bad = FakeSource(RuntimeError)
    monkeypatch.setattr(fetch, "MARKETS", {"TWSE": ok, "TPEx": bad})
    day = dt.date(2026, 11, 5)
    fetch.failures.clear()
    fetch.fetch_optional(day)
    assert store.has("margin", day, "TWSE") and not store.has("margin", day, "TPEx")
    assert fetch.failures == []                     # 選用資料失敗不算進失敗清單
    fetch.fetch_optional(day)
    assert ok.calls == 1 and bad.calls == 2         # 已存的不重抓，失敗的下次再試
    assert not store.is_complete(day)               # 是否完整只看必要資料


def test_update_holders_weekly(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    from screener.sources import tdcc
    calls = []
    monkeypatch.setattr(tdcc, "holders_latest", lambda: calls.append(1) or parse.holders_table(_tdcc_csv()))
    monkeypatch.setattr(fetch, "today_taipei", lambda: dt.date(2026, 11, 5))
    assert fetch.update_holders() == "2026-10-30"
    assert store.holders_days() == [dt.date(2026, 10, 30)]
    assert fetch.update_holders() is None and len(calls) == 1     # 不到 8 天不重抓
    monkeypatch.setattr(fetch, "today_taipei", lambda: dt.date(2026, 11, 7))
    fetch.update_holders()
    assert len(calls) == 2
    h = store.load(store.HOLDERS)
    assert h["date"].iloc[0] == pd.Timestamp("2026-10-30") and h["code"].iloc[0] == "2330"


def test_institutional_old_tpex_headers():
    """2018-01 以前櫃買中心的三大法人表：欄名用「淨買」（GitHub Actions 回補 2013～2018 年時失敗的格式）。"""
    fields = ["代號", "名稱", "外資及陸資買進股數", "外資及陸資賣股數", "外資及陸資淨買股數", "投信買進股數", "投信賣股數",
              "投信淨買股數", "自營淨買股數", "自營商(自行買賣)買股數", "自營商(自行買賣)賣股數", "自營商(自行買賣)淨買股數",
              "自營商(避險)買股數", "自營商(避險)賣股數", "自營商(避險)淨買股數", "三大法人買賣超股數"]
    rows = [["6488", "環球晶", "1,000", "400", "600", "50", "20", "30", "-70", "10", "50", "-40", "0", "30", "-30", "560"]]
    r = parse.institutional_table(fields, rows).iloc[0]
    assert (r.code, r.foreign_net, r.trust_net, r.dealer_net, r.total_net) == ("6488", 600, 30, -70, 560)


def test_old_days_skip_unavailable_sources(tmp_path, monkeypatch):
    """2006 年：上櫃股價、三大法人都還沒有，不該送請求，那天也算完整。"""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    calls = []

    class Src:
        def __init__(self, market):
            self.market = market

        def prices(self, day):
            calls.append((self.market, "prices"))
            return pd.DataFrame({"code": ["2330"], "name": ["台積電"], "market": [self.market], "open": [1.0],
                                 "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0], "value": [1.0]})

        def institutional(self, day):
            calls.append((self.market, "institutional"))
            return None

        def margin(self, day):
            calls.append((self.market, "margin"))
            return parse.margin_table(TWSE_FIELDS, TWSE_ROWS)

    monkeypatch.setattr(fetch, "MARKETS", {"TWSE": Src("TWSE"), "TPEx": Src("TPEx")})
    day = dt.date(2006, 12, 28)
    assert fetch.fetch_day(day) == "complete"
    assert calls == [("TWSE", "prices"), ("TWSE", "margin")]       # 上櫃全部、上市三大法人都沒去抓
    calls.clear()
    assert fetch.fetch_day(day) == "complete" and calls == []        # 重跑不再送任何請求
    assert store.available("institutional", dt.date(2014, 12, 1), "TPEx")
    assert not store.available("institutional", dt.date(2014, 11, 28), "TPEx")
