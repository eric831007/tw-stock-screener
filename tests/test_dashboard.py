import pytest
import json
import re

import numpy as np
import pandas as pd

from screener import dashboard, store


def test_build_dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "data")
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2025-01-02", periods=300)
    stocks = {"2330": ("台積電", 1000), "2317": ("鴻海", 200), "0050": ("元大台灣50", 100)}
    paths = {k: 100 + np.cumsum(rng.normal(0, 1, len(days))) for k in stocks}
    for i, d in enumerate(days):
        rows = []
        for code, (name, scale) in stocks.items():
            c = paths[code][i] * scale / 100
            rows.append({"code": code, "name": name, "market": "TWSE", "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1e6, "value": c * 1e6})
        store.save("prices", d.date(), "TWSE", pd.DataFrame(rows))
    store.save_stock_info(pd.DataFrame({"code": ["2330", "2317"], "name": ["台積電", "鴻海"], "market": "TWSE",
                                        "industry_code": ["24", "31"], "shares": [2.6e10, 1.4e10]}))
    monkeypatch.setattr(dashboard, "load_watchlist", lambda: {"top_market_cap": 2, "my_etfs": ["0050"]})
    out = dashboard.build(tmp_path / "index.html")
    html = out.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert data["basis"] == "市值"
    assert [s["code"] for s in data["stocks"]] == ["0050", "2330", "2317"]
    s = data["stocks"][1]
    assert len(s["D"]["t"]) == 300 and s["D"]["ma240"][-1] is not None
    assert len(s["W"]["t"]) == 61 and len(s["M"]["t"]) == 14
    assert set(s["bt"]) >= {"rsi_rebound", "_any_day"}
    assert s["verdict"]["label"] in {"強多", "偏多", "中性", "偏空", "強空"}
    assert len(s["verdict"]["checks"]) == 6 and len(s["D"]["v"]) == 300
    assert s["action"]["code"] in {"buy", "add", "half", "sell", "hold", "flat"} and s["action"]["lines"]
    for r in s["rounds"]:
        assert r["acts"][0]["t"] == "buy" and (r["e"] is None or r["e"] > r["s"])
    assert set(s["modeStats"]) == {"winrate", "all_in", "scale", "auto"}
    assert data["modePooled"]["all_in"]["n"] == sum(x["modeStats"]["all_in"]["n"] for x in data["stocks"])
    assert data["trading"]["entry"] and s["regime"] in {"bull", "bear", "range"}   # 300 天 ≥ 260 天，算得出牛熊
    assert all(x["regime"] == data["stocks"][0]["regime"] for x in data["stocks"])  # 全部用 0050 的牛熊
    assert len(s["D"]["rg"]) == 300


def test_empty_verdict_has_all_fields():
    v = dashboard._verdict_now(pd.DataFrame(columns=["level"]), pd.DataFrame())
    assert {"buys", "sells", "buyPts", "sellPts", "checks"} <= set(v)


def test_px_handles_nan():
    assert dashboard._px(float("nan")) == "—" and dashboard._px(1234.5) == "1,234.5"


def test_regime_fallback_when_market_code_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "data")
    days = pd.bdate_range("2025-01-02", periods=30)
    for i, d in enumerate(days):
        store.save("prices", d.date(), "TWSE", pd.DataFrame([{"code": "2330", "name": "台積電", "market": "TWSE",
            "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i, "volume": 1e6, "value": 1e8}]))
    store.save_stock_info(pd.DataFrame({"code": ["2330"], "name": ["台積電"], "market": "TWSE",
                                        "industry_code": ["24"], "shares": [2.6e10]}))
    monkeypatch.setattr(dashboard, "load_watchlist", lambda: {"top_market_cap": 1, "my_etfs": []})
    monkeypatch.setattr(dashboard, "load_trading", lambda: dashboard.sg.TradeSettings(regime_from="9999"))
    html = dashboard.build(tmp_path / "i.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert data["trading"]["regimeFrom"] == "self" and "9999" in data["trading"]["regimeNote"]
    assert data["holdAvg"] == pytest.approx(data["stocks"][0]["hold"])


def test_build_stock_handles_data_gap():
    a = pd.bdate_range("2025-01-02", periods=120)
    b = pd.bdate_range("2025-09-01", periods=80)
    rng = np.random.default_rng(2)
    c = 100 + np.cumsum(rng.normal(0, 1, 200))
    daily = pd.DataFrame({"date": list(a) + list(b), "open": c, "high": c * 1.01, "low": c * 0.99,
                          "close": c, "volume": 1e6})
    out = dashboard.build_stock(daily, code="2330")
    assert out["gaps"] == [120] and len(out["D"]["t"]) == 200
    assert out["D"]["ma60"][120 + 58] is None and out["D"]["ma60"][120 + 59] is not None   # 缺口後重新累積
    for r in out["rounds"]:                                 # 交易不會跨過缺口
        assert (r["s"] < 120) == (r["e"] is None or r["e"] < 120) or r["e"] is None
