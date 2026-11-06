import numpy as np
import pandas as pd
import pytest

from screener import indicators as ind
from screener import signals as sg

NO_MA = sg.TradeSettings(ma_entry=False)


def frame(closes):
    dates = pd.bdate_range("2025-01-01", periods=len(closes))
    c = pd.Series(closes, dtype=float)
    return ind.add_all(pd.DataFrame({"date": dates, "open": c, "high": c * 1.005, "low": c * 0.995,
                                     "close": c, "volume": 1000.0}))


def test_rsi_rebound_fires_after_selloff():
    closes = [100.0] * 30 + [100 - 2 * i for i in range(1, 16)] + [72 + 3 * i for i in range(1, 6)]
    df = frame(closes)
    s = sg.detect(df)
    fired = df.index[s["rsi_rebound"]]
    assert len(fired) == 1 and fired[0] >= 45  # only on the bounce, not during the fall


def test_pullback_needs_long_uptrend():
    up = list(np.linspace(50, 150, 300))
    dip = [150 - i * 2.5 for i in range(1, 8)]
    df = frame(up + dip + [dip[-1] + 2] * 3)
    s = sg.detect(df)
    assert not s["pullback_ma60"].iloc[:240].any()  # no MA240 yet -> never fires
    assert list(df.index[s["pullback_ma60"]]) == [302]  # the day the low touches MA60 and closes above it


def test_backtest_and_baseline():
    rng = np.random.default_rng(3)
    df = frame(list(100 + np.cumsum(rng.normal(0.05, 1.5, 600))))
    s = sg.detect(df)
    bt = sg.backtest(df, s)
    assert bt["_any_day"]["n20"] == 600 - 20
    for name in sg.ALL_SIGNALS:
        assert bt[name]["count"] == int(s[name].sum())
        assert bt[name]["n60"] <= bt[name]["count"]
    pooled = sg.pool_backtests([bt, bt])
    assert pooled["_any_day"]["n20"] == 2 * (600 - 20)
    assert pooled["_any_day"]["avg20"] == pytest.approx(bt["_any_day"]["avg20"])


def test_buy_score_counts_distinct_signals():
    sig = pd.DataFrame({k: [False] * 8 for k in sg.ALL_SIGNALS})
    sig.loc[2, "rsi_rebound"] = True
    sig.loc[4, "kd_cross_low"] = True
    sig.loc[5, "kd_cross_low"] = True
    score = sg.buy_score(sig)
    assert list(score) == [0, 0, 1, 1, 2, 2, 2, 1]


def test_sell_signals():
    up = list(np.linspace(50, 150, 300))
    drop = [150 - i * 4 for i in range(1, 12)]
    df = frame(up + drop)
    s = sg.detect(df)
    i60 = df.index[s["break_ma60"]]
    assert len(i60) == 1 and i60[0] > 300  # crosses below MA60 once, during the drop
    assert s["rsi_fall"].iloc[300:].any()  # leaves overbought as the drop starts
    assert not s["break_ma60"].iloc[:300].any()
    assert sg.sell_score(s).iloc[-1] >= 1
    assert sg.buy_score(s).max() <= len(sg.SIGNALS)


def test_verdict_levels_follow_trend():
    up = list(np.linspace(50, 150, 300))
    down = list(np.linspace(150, 60, 120))
    df = frame(up + down)
    v = sg.verdict(df, sg.detect(df))
    assert v["total"].between(-8, 8).all()
    assert (v["level"].iloc[250:300] >= 0).all()   # steady uptrend: never bearish
    assert v["level"].iloc[-1] <= 0               # long decline: not bullish at the end
    assert set(v["level"].unique()) <= set(sg.VERDICTS)
    assert v.loc[v["capped"] == "過熱", "level"].eq(0).all()


def test_verdict_overheat_caps_buy():
    closes = [100.0] * 80 + [100 * 1.06 ** i for i in range(1, 8)]
    df = frame(closes)
    v = sg.verdict(df, sg.detect(df))
    assert v["capped"].iloc[-1] == "過熱" and v["level"].iloc[-1] == 0


def _quiet(df):
    """沒有任何訊號的 sig，讓測試自己放訊號。"""
    sig = sg.detect(df)
    for k in sig:
        sig[k] = False
    return sig


def test_simulate_buys_after_bottom_and_sells_on_weakness():
    # 下跌 → 底部反彈（買）→ 上漲 → 轉弱跌破 10 日線（賣）
    closes = ([100 - i for i in range(40)] + [61 + 1.5 * i for i in range(30)] + [105 - 3 * i for i in range(12)])
    df = frame(closes)
    sig = sg.detect(df)
    sim = sg.simulate(df, sig, "all_in")
    closed = [r for r in sim["rounds"] if r["end_i"] is not None]
    assert closed, "should complete at least one round"
    first = closed[0]
    buy = first["actions"][0]
    assert buy["type"] == "buy" and 40 <= buy["i"] < 55      # 在底部附近買，不是漲完才買
    assert first["actions"][-1]["type"] == "exit" and first["ret"] > 0.2
    assert sg.entry_signal(df, sig).iat[buy["i"] - 1]         # 成交日 = 訊號隔天
    assert buy["px"] == df["open"].iat[buy["i"]]
    assert all(a["type"] in ("buy", "exit") for r in sim["rounds"] for a in r["actions"])  # 全進全出不會加碼或賣一半
    for a, b in zip(sim["rounds"], sim["rounds"][1:]):         # 一輪賣完才開始下一輪
        assert b["start_i"] > a["end_i"]


def test_simulate_stop_loss_and_pending():
    df = frame([100.0] * 60)
    sig = _quiet(df)
    sig.loc[50, "rsi_rebound"] = True
    df.loc[51:, "k"], df.loc[51:, "d"] = 60.0, 40.0
    df.loc[51, "close"] = 101.0
    df.loc[54:, ["open", "close"]] = 90.0                     # 成本 100 → 跌 10%，觸發停損
    r = sg.simulate(df, sig, "all_in", settings=NO_MA)["rounds"][0]
    assert r["start_i"] == 52 and r["end_i"] == 55 and r["actions"][-1]["reason"] == "stop_loss"
    st = sg.round_stats([r])
    assert st["n"] == 1 and st["avg"] < 0

    df2 = frame([100.0] * 60)
    sig2 = _quiet(df2)
    sig2.loc[57, "rsi_rebound"] = True
    df2.loc[59, ["k", "d", "close"]] = [60.0, 40.0, 101.0]
    assert sg.simulate(df2, sig2, "all_in", settings=NO_MA)["pending"]["type"] == "buy"   # 最後一天才出現 → 下一個交易日開盤買


def test_scale_adds_at_lows_and_sells_half_at_highs():
    df = frame([100.0] * 80)
    df[["ma10", "ma5"]] = 90.0                                 # 收盤一直在 10 日線、5 日線之上 → 不會因跌破而清倉
    df["k"], df["d"] = 60.0, 40.0
    sig = _quiet(df)
    sig.loc[50, "rsi_rebound"] = True                          # 底部確認 → 51 買
    df.loc[55, "k"] = 30.0                                     # 底部確認中斷
    sig.loc[60, "kd_cross_low"] = True                         # 再次成立 → 61 加碼
    sig.loc[65, "kd_cross_high"] = True                        # 區域高點 → 66 賣一半
    sim = sg.simulate(df, sig, "scale", settings=NO_MA)
    acts = [(a["i"], a["type"], a["units"]) for a in sim["rounds"][0]["actions"]]
    assert acts[:3] == [(51, "buy", 1), (61, "add", 2), (66, "half", 1)]   # 賣一半：2 份 → 1 份
    assert sim["position"]["units"] == 1 and sim["rounds"][0]["end_i"] is None
    assert sg.simulate(df, sig, "all_in", settings=NO_MA)["rounds"][0]["actions"][-1]["type"] == "exit"  # 同樣情況全進全出直接清倉


def test_scale_respects_max_units():
    df = frame([100.0] * 80)
    df[["ma10", "ma5"]] = 90.0
    df["k"], df["d"] = 60.0, 40.0
    sig = _quiet(df)
    for i, day in enumerate((40, 45, 50, 55, 60)):             # 5 次底部確認，每次中間都中斷
        sig.loc[day, "rsi_rebound"] = True
        df.loc[day - 1, "k"] = 30.0
    sim = sg.simulate(df, sig, "scale", settings=sg.TradeSettings(ma_entry=False, max_units=3))
    assert max(a["units"] for a in sim["rounds"][0]["actions"]) == 3


def test_regime_and_auto_mode():
    up = list(np.linspace(50, 150, 320))
    df = frame(up)
    rg = sg.regime(df)
    assert (rg.iloc[:259] == "unknown").all() and rg.iloc[-1] == "bull"
    down = frame(list(np.linspace(150, 50, 320)))
    assert sg.regime(down).iloc[-1] == "bear"
    sig = sg.detect(df)
    bull = pd.Series("bull", index=df.index)
    bear = pd.Series("bear", index=df.index)
    a = sg.simulate(df, sig, "auto", bull)
    b = sg.simulate(df, sig, "scale", bull)
    assert [r["actions"] for r in a["rounds"]] == [r["actions"] for r in b["rounds"]]
    c = sg.simulate(df, sig, "auto", bear)
    d = sg.simulate(df, sig, "all_in", bear)
    assert [r["actions"] for r in c["rounds"]] == [r["actions"] for r in d["rounds"]]
    with pytest.raises(ValueError):
        sg.simulate(df, sig, "auto")


def test_ma_crosses_and_alignment():
    closes = [100 - i for i in range(40)] + [61 + 2 * i for i in range(40)]
    df = frame(closes)
    sig = sg.detect(df)
    up = df.index[sig["ma5x10_up"]]
    assert len(up) == 1 and 40 < up[0] < 50                 # 反轉後 5 日線上穿 10 日線一次
    assert not sig["ma5x10_down"].iloc[45:].any()
    al = sg.ma_alignment(df)
    assert al.iloc[30] == "bear" and al.iloc[-1] == "bull" and al.iloc[0] == "unknown"
    # 不算進底部訊號：只有均線交叉時，buy_score 不變
    assert sg.buy_score(sig).equals(sg.buy_score(sig.assign(ma5x10_up=False, ma10x20_up=False, ma20x60_up=False)))


def test_ma_entry_filter_blocks_buy_while_falling():
    df = frame([100.0] * 60)
    df["k"], df["d"] = 60.0, 40.0
    df["ma5"] = 99.0                                         # 收盤站上 5 日線
    df["ma10"] = np.linspace(110, 100, 60)                 # 10 日線一路往下：下跌還沒結束
    sig = _quiet(df)
    sig.loc[50, "rsi_rebound"] = True
    assert sg.simulate(df, sig, "all_in", settings=sg.TradeSettings(ma_entry=True))["rounds"] == []
    assert sg.simulate(df, sig, "all_in", settings=NO_MA)["rounds"]


def test_ma_hold_keeps_position_in_uptrend():
    df = frame([100.0] * 80)
    df["ma5"], df["ma20"] = 95.0, 85.0
    df["ma10"] = np.linspace(88, 94, 80)                   # 10 日線往上且在月線之上，收盤在 10 日線之上
    df["k"], df["d"] = 60.0, 40.0
    sig = _quiet(df)
    sig.loc[50, "rsi_rebound"] = True
    sig.loc[60, "kd_cross_high"] = True
    held = sg.simulate(df, sig, "all_in", settings=sg.TradeSettings(ma_hold=True))
    sold = sg.simulate(df, sig, "all_in", settings=sg.TradeSettings(ma_hold=False))
    assert held["rounds"][0]["end_i"] is None
    assert sold["rounds"][0]["end_i"] == 61


def test_half_sell_frees_capacity_for_adds():
    df = frame([100.0] * 90)
    df[["ma10", "ma5"]] = 90.0
    df["k"], df["d"] = 60.0, 40.0
    sig = _quiet(df)
    for day in (40, 45, 50):                                    # 買 + 加 + 加 → 3 份
        sig.loc[day, "rsi_rebound"] = True
        df.loc[day - 1, "k"] = 30.0
    sig.loc[55, "kd_cross_high"] = True                         # 賣一半 → 1.5 份
    sig.loc[60, "rsi_rebound"] = True                           # 新低點 → 可以再加 1 份（2.5 ≤ 3）
    df.loc[59, "k"] = 30.0
    acts = [(a["type"], a["units"]) for a in sg.simulate(df, sig, "scale", settings=NO_MA)["rounds"][0]["actions"]]
    assert acts == [("buy", 1), ("add", 2), ("add", 3), ("half", 1.5), ("add", 2.5)]


def test_no_rebuy_on_stale_bottom_signal():
    df = frame([100.0] * 60)
    df["k"], df["d"] = 60.0, 40.0
    df["ma5"], df["ma10"] = 99.0, 90.0
    sig = _quiet(df)
    sig.loc[40, "rsi_rebound"] = True                           # 41 買進
    df.loc[42, "ma10"] = 101.0                                  # 42 收盤跌破 10 日線 → 43 賣出
    rounds = sg.simulate(df, sig, "all_in", settings=NO_MA)["rounds"]
    assert len(rounds) == 1 and rounds[0]["end_i"] == 43        # 40 的舊訊號還在 5 日內，但不會再買回
    sig.loc[50, "kd_cross_low"] = True                          # 新的底部訊號 → 才會再買
    rounds = sg.simulate(df, sig, "all_in", settings=NO_MA)["rounds"]
    assert len(rounds) == 2 and rounds[1]["start_i"] == 51


def test_mode_for():
    assert sg.mode_for("auto", "bull") == "scale"
    assert all(sg.mode_for("auto", r) == "all_in" for r in ("bear", "range", "unknown"))
    assert sg.mode_for("all_in", "bull") == "all_in"


def _win_frame(n=80):
    df = frame([100.0] * n)
    df["ma60"], df["rsi3"] = 90.0, 50.0                         # 收盤在季線之上，RSI(3) 中性
    return df


def test_winrate_buys_dip_and_sells_on_rebound():
    df = _win_frame()
    df.loc[65, "rsi3"] = 20.0                                   # 短線急跌 → 66 開盤買
    df.loc[68, "rsi3"] = 75.0                                   # 反彈過熱 → 69 開盤賣
    r = sg.simulate(df, _quiet(df), "winrate")["rounds"]
    assert [(a["i"], a["type"], a["reason"]) for a in r[0]["actions"]] == [
        (66, "buy", "rsi3_low"), (69, "exit", "rsi3_high")]


def test_winrate_needs_trend_and_has_time_stop():
    df = _win_frame()
    df["ma60"] = 110.0                                          # 收盤在季線之下 → 不買
    df.loc[40, "rsi3"] = 10.0
    assert sg.simulate(df, _quiet(df), "winrate")["rounds"] == []
    df = _win_frame()
    df.loc[40, "rsi3"] = 10.0                                   # 41 買，之後 RSI 一直沒過熱
    r = sg.simulate(df, _quiet(df), "winrate")["rounds"][0]
    assert r["actions"][-1]["reason"] == "max_days" and r["end_i"] == 41 + sg.WIN_MAX_DAYS


def test_costs_are_deducted():
    df = _win_frame()
    df.loc[65, "rsi3"], df.loc[68, "rsi3"] = 20.0, 75.0          # 開盤價都是 100：毛報酬 0
    gross = sg.simulate(df, _quiet(df), "winrate", settings=sg.TradeSettings(fee=0))["rounds"][0]["ret"]
    net = sg.simulate(df, _quiet(df), "winrate", tax=sg.tax_for("2330"))["rounds"][0]["ret"]
    assert gross == pytest.approx(0) and net == pytest.approx(1 / 1.001425 * (1 - 0.001425 - 0.003) - 1)
    assert sg.tax_for("0050") == 0.001 and sg.tax_for("2330") == 0.003
