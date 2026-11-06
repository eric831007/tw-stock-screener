import numpy as np
import pandas as pd
import pytest

from screener import indicators as ind


def bars(closes, start="2026-01-05"):
    dates = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, dtype=float)
    return pd.DataFrame({"date": dates, "open": c, "high": c + 1, "low": c - 1, "close": c,
                         "volume": 1000.0})


def wilder_rsi_loop(closes, n=14):
    """Reference implementation written the textbook way."""
    out = [np.nan] * len(closes)
    gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, len(closes))]
    ag = al = None
    for i in range(1, len(closes)):
        g, l = gains[i - 1], losses[i - 1]
        if i < n:
            continue
        if ag is None:
            ag, al = np.mean(gains[:n]), np.mean(losses[:n])
        else:
            ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l) / n
        out[i] = 100 - 100 / (1 + ag / al) if al else 100.0
    return out


def test_rsi_close_to_textbook():
    rng = np.random.default_rng(1)
    closes = list(100 + np.cumsum(rng.normal(0, 1, 300)))
    ours = ind.rsi(pd.Series(closes)).to_numpy()
    ref = np.array(wilder_rsi_loop(closes))
    # same SMA seed as the textbook, so values match from the first RSI
    assert np.nanmax(np.abs(ours - ref)) < 1e-9 and np.isnan(ours[:14]).all()
    assert ((ours[20:] >= 0) & (ours[20:] <= 100)).all()


def test_rsi_all_up_is_100():
    r = ind.rsi(pd.Series(range(1, 40), dtype=float))
    assert r.iloc[-1] == pytest.approx(100)


def test_macd_matches_definition():
    closes = pd.Series(np.linspace(100, 130, 60))
    m = ind.macd(closes)
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    assert m["dif"].iloc[-1] == pytest.approx(ema12.iloc[-1] - ema26.iloc[-1])
    assert m["hist"].iloc[-1] == pytest.approx(m["dif"].iloc[-1] - m["dea"].iloc[-1])
    assert m["dif"].iloc[:25].isna().all()


def test_kd_taiwan_formula():
    b = bars([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    k = ind.kd(b["high"], b["low"], b["close"])
    # day 9: lowest low=9, highest high=19, close=18 -> RSV=90; K=50*2/3+90/3=63.33, D=50*2/3+63.33/3=54.44
    assert k["k"].iloc[8] == pytest.approx(63.333, abs=0.01)
    assert k["d"].iloc[8] == pytest.approx(54.444, abs=0.01)
    assert k["k"].iloc[:8].isna().all()


def test_weekly_and_monthly_resample():
    b = bars(list(range(1, 31)), start="2026-03-02")  # Mon 2 Mar .. Fri 10 Apr
    w = ind.resample(b, "W")
    assert len(w) == 6
    first = w.iloc[0]
    assert (first["open"], first["close"], first["high"], first["low"], first["volume"]) == (1, 5, 6, 0, 5000)
    assert first["date"] == pd.Timestamp("2026-03-06")
    m = ind.resample(b, "M")
    assert list(m["date"]) == [pd.Timestamp("2026-03-31"), pd.Timestamp("2026-04-10")]


def test_add_all_columns():
    df = ind.add_all(bars(list(np.linspace(50, 80, 260))))
    for c in ["ma5", "ma240", "rsi", "dif", "dea", "hist", "k", "d", "bb_up", "bb_low", "vol_ma20"]:
        assert c in df
    assert df["ma240"].iloc[-1] == pytest.approx(df["close"].iloc[-240:].mean())


def test_adjust_splits_restores_continuity():
    dates = pd.bdate_range("2025-06-01", periods=6)
    close = [196.0, 195.0, 49.0, 49.5, 50.0, 50.5]              # 1 拆 4
    df = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close, "volume": 1000.0})
    adj = ind.adjust_splits(df)
    assert adj["close"].iloc[1] == pytest.approx(195 * 49 / 195)
    assert (adj["close"].pct_change().abs().dropna() < 0.1).all()
    assert adj["volume"].iloc[0] == pytest.approx(1000 * 195 / 49)
    small = df.assign(open=[100, 95, 96, 97, 98, 99.0], close=[100, 95, 96, 97, 98, 99.0])
    assert ind.adjust_splits(small)["close"].tolist() == small["close"].tolist()   # 一般漲跌不動


def test_rsi3_column():
    df = ind.add_all(pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=30), "open": 1.0, "high": 1.0,
                                   "low": 1.0, "close": list(range(1, 31)), "volume": 1.0}))
    assert df["rsi3"].iloc[-1] == 100.0
