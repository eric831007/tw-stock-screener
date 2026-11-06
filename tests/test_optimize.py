import numpy as np
import pandas as pd

from screener import optimize as opt
from screener import signals as sg


def bars(dates, closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"date": pd.to_datetime(dates), "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": 1000.0})


def test_segments_split_on_gap_and_adjust_each_part():
    a = pd.bdate_range("2025-01-01", periods=60)
    b = pd.bdate_range("2025-09-01", periods=40)              # 中間缺好幾個月
    daily = bars(list(a) + list(b), [100.0] * 60 + [300.0] * 40)
    segs = opt.segments(daily)
    assert [len(d) for d, _ in segs] == [60, 40]
    assert segs[1][0]["close"].iloc[0] == 300.0               # 缺口前後的價差不會被當成拆股還原
    assert "rsi2" in segs[0][0] and "kd_cross_high" in segs[0][1]
    assert opt.segments(bars(pd.bdate_range("2025-01-01", periods=10), [1.0] * 10)) == []   # 太短的段丟掉


def test_optimize_stock_picks_train_only_and_judges_on_test():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-02", periods=700)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(days))))
    segs = opt.segments(bars(days, closes))
    split = days[450]
    grid = opt.candidates()[:40]
    r = opt.optimize_stock(segs, "2330", split, grid)
    assert r["verdict"] in {"適用", "不適用", "樣本不足", "找不到"}
    if r["best"] is not None:
        assert r["best"] in grid
        assert r["train"]["n"] >= opt.MIN_TRADES and r["train"]["avg"] > 0
        # 挑選期確實是所有候選裡勝率最高的（只看挑選期）
        for st in grid:
            s = opt.summarize(opt._split(opt.trades(segs, st, sg.tax_for("2330")), split)[0])
            if s["n"] >= opt.MIN_TRADES and s["avg"] > 0:
                assert (s["win"], s["avg"]) <= (r["train"]["win"], r["train"]["avg"])


def test_verdict_rules():
    assert len(opt.candidates()) == np.prod([len(v) for v in opt.GRID.values()])
    assert all(c.win_osc in ("rsi2", "rsi3") for c in opt.candidates())
    assert opt.summarize([]) == {"n": 0, "win": None, "avg": None, "worst": None}
    s = opt.summarize([0.1, -0.05, 0.02])
    assert s["n"] == 3 and abs(s["win"] - 2 / 3) < 1e-9 and s["worst"] == -0.05


def test_winrate_rule_text():
    st = sg.TradeSettings(win_trend="none", win_osc="rsi2", win_buy_below=10, win_sell_above=80, win_max_days=10)
    assert sg.winrate_rule(st) == "買進：RSI(2) 低於 10；賣出：RSI(2) 高於 80，或持有滿 10 個交易日"
    assert "收盤在季線之上" in sg.WINRATE_RULE
