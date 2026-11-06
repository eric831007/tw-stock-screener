"""技術指標：均線、RSI、MACD、KD、布林通道，以及日線轉週線／月線。

輸入是一檔股票依日期排序的 DataFrame，欄位：date, open, high, low, close, volume。
所有公式採台灣看盤軟體常用的版本（KD 用 9 日 RSV、1/3 平滑，起始值 50）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MA_PERIODS = {"D": (5, 10, 20, 60, 120, 240), "W": (5, 10, 20, 52), "M": (3, 6, 12, 24)}


def resample(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """日線轉週線（freq='W'）或月線（freq='M'）。日期用該期間最後一個交易日。"""
    if freq == "D":
        return daily.reset_index(drop=True)
    rule = {"W": "W-FRI", "M": "ME"}[freq]
    d = daily.dropna(subset=["close"]).set_index("date", drop=False)
    try:
        g = d.resample(rule)
    except ValueError:  # pandas < 2.2 還不認得 "ME"，用舊寫法 "M"
        g = d.resample("M")
    out = pd.DataFrame({
        "date": g["date"].max(),  # 該週／該月最後一個交易日
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })
    return out.dropna(subset=["close"]).reset_index(drop=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI：前 n 天用簡單平均起算，之後用 (前值×(n-1)+今日)/n 平滑。"""
    diff = close.diff().to_numpy()
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return pd.Series(out, index=close.index)
    gains, losses = np.clip(diff, 0, None), np.clip(-diff, 0, None)
    ag, al = gains[1:n + 1].mean(), losses[1:n + 1].mean()
    for i in range(n, len(close)):
        if i > n:
            ag = (ag * (n - 1) + gains[i]) / n
            al = (al * (n - 1) + losses[i]) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return pd.Series(out, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    out = pd.DataFrame({"dif": dif, "dea": dea, "hist": dif - dea})
    out.iloc[: slow - 1] = np.nan  # 前面資料不足時不顯示
    return out


def kd(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9) -> pd.DataFrame:
    lowest, highest = low.rolling(n).min(), high.rolling(n).max()
    rsv = ((close - lowest) / (highest - lowest).replace(0, np.nan) * 100).fillna(50)
    k_vals, d_vals, k, d = [], [], 50.0, 50.0
    for i, v in enumerate(rsv):
        if i < n - 1:
            k_vals.append(np.nan); d_vals.append(np.nan); continue
        k = k * 2 / 3 + v / 3
        d = d * 2 / 3 + k / 3
        k_vals.append(k); d_vals.append(d)
    return pd.DataFrame({"k": k_vals, "d": d_vals}, index=close.index)


def bollinger(close: pd.Series, n: int = 20, width: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_up": mid + width * sd, "bb_low": mid - width * sd})


def adjust_splits(daily: pd.DataFrame, lo: float = 0.85, hi: float = 1.18) -> pd.DataFrame:
    """還原分割／合併造成的價格跳動（例如 0050 在 2025 年 1 拆 4，股價從 195 變 49）。

    台股單日漲跌幅限制 ±10%，開盤價對前一天收盤若超出 lo～hi，就判斷是股本變動，
    把之前的價格乘上這個比例、成交量除以這個比例，讓整段走勢連續。除權息的小缺口不處理。
    """
    d = daily.reset_index(drop=True).copy()
    ratio = d["open"] / d["close"].shift(1)
    for i in d.index[(ratio < lo) | (ratio > hi)]:
        f = ratio.iat[i]
        d.loc[: i - 1, ["open", "high", "low", "close"]] *= f
        d.loc[: i - 1, "volume"] /= f
    return d


GAP_DAYS = 15


def split_gaps(daily: pd.DataFrame, gap_days: int = GAP_DAYS) -> list[pd.DataFrame]:
    """資料有缺口（相鄰兩個交易日差超過 gap_days 天）時切成好幾段，每段各自還原拆股。

    指標要在每段裡各自計算，避免均線、RSI 等跨過缺口把兩段不相干的價格混在一起。
    """
    if daily.empty:
        return []
    daily = daily.reset_index(drop=True)
    gaps = daily["date"].diff().dt.days.fillna(0) > gap_days
    return [adjust_splits(seg) if len(seg) > 1 else seg.reset_index(drop=True)
            for _, seg in daily.groupby(gaps.cumsum())]


def add_all(bars: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """在 K 線資料上加上所有指標欄位。"""
    df = bars.copy().reset_index(drop=True)
    for p in MA_PERIODS[freq]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["rsi"] = rsi(df["close"])
    df["rsi2"] = rsi(df["close"], 2)  # 短線超買超賣（高勝率模式與個股最佳化用）
    df["rsi3"] = rsi(df["close"], 3)
    df = df.join(macd(df["close"])).join(kd(df["high"], df["low"], df["close"])).join(bollinger(df["close"]))
    return df
