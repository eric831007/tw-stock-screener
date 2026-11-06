import datetime as dt

import pandas as pd
import pytest

from screener import userdata as ud


@pytest.fixture
def tmp_mydata(tmp_path, monkeypatch):
    monkeypatch.setattr(ud, "MYDATA", tmp_path / "mydata")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("watchlist:\n  my_etfs: [\"0050\"]\n  extra: []                   # 想額外加入的股票代號\n"
                   "trading:\n  mode: winrate\n", encoding="utf-8")
    monkeypatch.setattr(ud, "CONFIG", cfg)
    return tmp_path


def test_read_trades_chinese_columns_and_merge(tmp_mydata):
    f = tmp_mydata / "broker.csv"
    pd.DataFrame({"成交日期": ["2026/10/14", "2026/10/20", "2026/10/14"], "股票代號": ["0050", "0050 元大台灣50", "2330"],
                  "買賣別": ["現買", "賣出", "買"], "成交價": ["113", "1,05.5", "2,240"], "成交股數": ["1,000", "500", "100"]}
                 ).to_csv(f, index=False, encoding="utf-8-sig")
    t = ud.read_trades(f)
    assert list(t["code"]) == ["0050", "2330", "0050"]
    assert list(t["side"]) == ["buy", "buy", "sell"]
    assert t["price"].tolist() == [113.0, 2240.0, 105.5] and t["shares"].tolist() == [1000, 100, 500]
    ud.import_trades(f)
    assert len(ud.import_trades(f)) == 3            # 重複匯入不會變多筆
    assert len(ud.load_trades()) == 3


def test_read_trades_reports_bad_rows(tmp_mydata):
    f = tmp_mydata / "bad.csv"
    pd.DataFrame({"日期": ["2026-10-14", "沒日期"], "代號": ["0050", "0050"], "買賣": ["買", "買"],
                  "價格": [113, 113], "股數": [1000, 1000]}).to_csv(f, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="第 3 列"):
        ud.read_trades(f)
    g = tmp_mydata / "missing.csv"
    pd.DataFrame({"日期": ["2026-10-14"], "代號": ["0050"]}).to_csv(g, index=False)
    with pytest.raises(ValueError, match="找不到欄位"):
        ud.read_trades(g)


def test_positions_fifo_with_costs():
    t = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-10"]), "code": ["2330"] * 3,
                      "side": ["buy", "buy", "sell"], "price": [1000.0, 1100.0, 1200.0], "shares": [100, 100, 100],
                      "fee": [None, None, None], "note": ""})
    p = ud.positions(t, {"2330": 1150.0}).iloc[0]
    buy1 = 1000 * 100 + 142                                   # 0.1425% = 142.5 → 無條件捨去 142 元
    sell = 1200 * 100 - 171 - 360                             # 手續費 171、證交稅 0.3% = 360
    assert p["shares"] == 100 and p["realized"] == pytest.approx(sell - buy1)
    assert p["avg_cost"] == pytest.approx((1100 * 100 + 156) / 100)       # 156.75 → 156 元
    assert ud.costs("buy", 10, 100, "0050") == ud.MIN_FEE     # 小額交易收最低 20 元
    assert ud.costs("sell", 100, 1000, "0050") == 142 + 100   # ETF 證交稅 0.1%


def test_compare_with_rule():
    t = pd.DataFrame({"date": pd.to_datetime(["2026-10-14", "2026-10-20", "2026-11-01"]), "code": ["0050"] * 3,
                      "side": ["buy", "sell", "buy"]})
    acts = {"0050": [(pd.Timestamp("2026-10-14"), "buy"), (pd.Timestamp("2026-10-21"), "buy")]}
    r = ud.compare_with_rule(t, acts).tolist()
    assert r[0].startswith("符合規則") and r[1].startswith("跟規則相反") and r[2] == "規則當時沒有訊號"


def test_screenshots(tmp_mydata):
    img = tmp_mydata / "shot.png"
    img.write_bytes(b"\x89PNG fake")
    a = ud.add_screenshot(img, "0050", dt.date(2026, 11, 5), "盤中急跌")
    b = ud.add_screenshot(img, "0050", dt.date(2026, 11, 5))
    assert a.exists() and b.exists() and a != b
    df = ud.list_screenshots("0050")
    assert len(df) == 2 and df["note"].iloc[0] == "盤中急跌" and df["date"].iloc[0] == "2026-11-05"
    assert ud.list_screenshots("2330").empty


def test_extra_codes_roundtrip(tmp_mydata):
    ud.set_extra_codes(["2412", "6488", "abc", "2412"])
    assert ud.extra_codes() == ["2412", "6488"]
    text = ud.CONFIG.read_text(encoding="utf-8")
    assert "# 想額外加入的股票代號" in text and "mode: winrate" in text
    ud.set_extra_codes([])
    assert ud.extra_codes() == []
