import numpy as np
import pandas as pd
import pytest

from screener import app_model, dashboard, store, userdata


@pytest.fixture
def env(tmp_path, monkeypatch):
    """假資料：兩檔 120 天，0050 有 2023 年的歷史放在額外資料夾。"""
    main, hist = tmp_path / "data", tmp_path / "history"
    monkeypatch.setattr(store, "DATA_DIR", main)
    monkeypatch.setattr(store, "EXTRA_DIRS", [hist])
    monkeypatch.setattr(app_model, "DEFAULT_HISTORY", tmp_path / "nope")
    monkeypatch.setattr(userdata, "MYDATA", tmp_path / "mydata")
    cfg = tmp_path / "config.yaml"
    cfg.write_text('watchlist:\n  top_market_cap: 1\n  my_etfs: ["0050"]\n  extra: []\n', encoding="utf-8")
    monkeypatch.setattr(userdata, "CONFIG", cfg)
    monkeypatch.setattr(dashboard, "load_watchlist", lambda: {"top_market_cap": 1, "my_etfs": ["0050"],
                                                               "extra": userdata.extra_codes()})
    rng = np.random.default_rng(5)
    stocks = {"0050": ("元大台灣50", 100.0), "2330": ("台積電", 1000.0), "2412": ("中華電", 120.0)}
    paths = {c: p0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 220))) for c, (_, p0) in stocks.items()}
    days = pd.bdate_range("2025-03-03", periods=220)
    old = pd.bdate_range("2023-01-02", periods=30)
    saved = store.DATA_DIR
    for i, d in enumerate(list(old) + list(days)):
        if i < 30:
            store.DATA_DIR = hist           # 前 30 天寫到歷史資料夾
            rows = [{"code": "0050", "name": "元大台灣50", "market": "TWSE", "open": 90, "high": 91, "low": 89,
                     "close": 90, "volume": 1e6, "value": 9e7}]
        else:
            store.DATA_DIR = main
            j = i - 30
            rows = [{"code": c, "name": n, "market": "TWSE", "open": paths[c][j], "high": paths[c][j] * 1.01,
                     "low": paths[c][j] * 0.99, "close": paths[c][j], "volume": 1e6, "value": paths[c][j] * 1e6}
                    for c, (n, _) in stocks.items() if not (c == "2412" and j == 219)]
        store.save("prices", d.date(), "TWSE", pd.DataFrame(rows))
    store.DATA_DIR = saved
    store.save_stock_info(pd.DataFrame({"code": ["2330", "2412"], "name": ["台積電", "中華電"], "market": "TWSE",
                                        "industry_code": ["24", "28"], "shares": [2.6e10, 7.7e9]}))
    return tmp_path


def test_model_loads_history_and_watchlist(env):
    m = app_model.Model()
    lo, hi = m.date_range
    assert lo == pd.Timestamp("2023-01-02") and hi == pd.Timestamp("2025-03-03") + pd.offsets.BDay(219)
    assert [w["code"] for w in m.watch] == ["0050", "2330"]
    s = m.analyze("0050")
    assert s.gaps == [30] and len(s.d) == 250             # 歷史段 + 近期段，中間是缺口
    assert all(r["start_i"] >= 30 or (r["end_i"] or 0) < 30 for r in s.sim["rounds"])
    assert s.action["label"] in {"續抱", "空手", "買進", "全部賣出", "加碼", "賣出一半"}
    summ = m.summary()
    assert list(summ["code"]) == ["0050", "2330"] and summ["close"].notna().all()


def test_add_and_remove_codes(env):
    m = app_model.Model()
    added, bad = m.add_codes(["2412", "9999"])
    assert added == ["2412"] and bad == ["9999"]
    assert "2412" in [w["code"] for w in m.watch]          # 最新一天沒成交也會列出
    assert userdata.extra_codes() == ["2412"]
    m.remove_code("2412")
    assert userdata.extra_codes() == [] and "2412" not in [w["code"] for w in m.watch]


def test_trades_compare_and_positions(env):
    m = app_model.Model()
    s = m.analyze("2330")
    f = env / "t.csv"
    first = s.d["date"].iat[100]
    pd.DataFrame({"日期": [first.strftime("%Y-%m-%d")], "代號": ["2330"], "買賣": ["買"],
                  "價格": [s.d["close"].iat[100]], "股數": [1000]}).to_csv(f, index=False, encoding="utf-8-sig")
    userdata.import_trades(f)
    t = m.my_trades()
    assert len(t) == 1 and isinstance(t["compare"].iat[0], str)
    p = m.my_positions().iloc[0]
    assert p["shares"] == 1000 and p["last"] == pytest.approx(s.last_close)


def test_desktop_window_smoke(env, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from screener import desktop
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    m = app_model.Model()
    w = desktop.MainWindow(m)
    w._list_ready(m.summary())
    w.current = "2330"
    w.show_stock("2330")
    assert w.table.rowCount() == 2 and "台積電" in w.header.text()
    assert w.rule_table.rowCount() == len(m.analyze("2330").sim["rounds"])
    w.close()


def test_chip_summary_streaks():
    d = pd.DataFrame({"volume": [1e6] * 6, "foreign_net": [-1e3, 2e3, 3e3, 1e3, 4e3, 5e3],
                      "trust_net": [0, 0, -1e3, -2e3, -3e3, -1e3], "dealer_net": [1e3, -1e3, 1e3, -1e3, 0, 0],
                      "total_net": [0.0] * 6})
    c = app_model.chip_summary(d)
    assert c["foreign_net"]["streak"] == 5 and c["trust_net"]["streak"] == -4 and c["dealer_net"]["streak"] == 0
    assert c["foreign_net"]["sum5"] == 15                     # 張
    lines = app_model.chip_lines(c)
    assert lines[0].startswith("外資：連續買超 5 天") and "連續賣超 4 天" in lines[1]
    assert app_model.chip_summary(pd.DataFrame({"volume": [1.0]})) == {}


def test_desktop_chart_timeframes_and_panes(env, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from screener import desktop
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    m = app_model.Model()
    c = desktop.Chart()
    c.show_stock(m.analyze("2330"))
    for tf in ("W", "M", "D"):
        c._set_tf(tf)
        assert len(c._f) > 0
    for key in desktop.PANES:
        c._toggle_pane(key, True)
    c._toggle_bb(True)
    assert "RSI(3)" in c.info.text() and "外資" in c.info.text()


def test_chart_defaults_and_remembered_settings(env, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from screener import desktop
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    c = desktop.Chart()
    assert c.panes_on == {"rule", "vol"} and c.ma_on["D"] == {5, 20, 60}     # 預設：RSI(3)、成交量；5日、月線、季線
    assert c.buys_on and not c.sells_on                                       # 預設只顯示買進標籤
    c._toggle_marks("sell", True)
    c._toggle_marks("buy", False)
    c._toggle_pane("macd", True)
    c._toggle_ma(240, True)
    c._toggle_bb(True)
    c2 = desktop.Chart()                                                      # 重新開啟：沿用上次的勾選
    assert c2.panes_on == {"rule", "vol", "macd"} and c2.ma_on["D"] == {5, 20, 60, 240} and c2.bb_on
    assert not c2.buys_on and c2.sells_on
    desktop.Chart.settings_path().write_text("壞掉的檔案", encoding="utf-8")
    assert desktop.Chart().panes_on == {"rule", "vol"}                       # 設定檔壞了就用預設
