"""買點／賣點訊號與歷史回測。

每個訊號都是「條件跟過去的低點／高點很像」的描述，不是保證會漲或會跌。
賣點訊號的回測看「之後下跌的比例」：比任意一天高，才代表這個賣點有用。
回測：找出過去每一次訊號出現的日子，統計之後 20 / 60 個交易日的報酬與上漲比例，
讓使用者知道這個訊號在這檔股票上的歷史表現與樣本數。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 訊號代碼 → (名稱, 說明)
SIGNALS = {
    "pullback_ma60": ("回測季線不破", "長線多頭（站上年線、季線在半年線之上）時，股價拉回碰到季線後收在季線之上"),
    "rsi_rebound": ("RSI 超賣回升", "RSI(14) 從 30 以下回到 30 以上"),
    "kd_cross_low": ("KD 低檔黃金交叉", "K 值在 25 以下由下往上穿過 D 值"),
    "macd_turn": ("MACD 綠柱縮短", "MACD 柱狀體仍為負，但連續 3 天縮短（跌勢減弱）"),
    "bb_rebound": ("布林下緣反彈", "前一天收在布林下軌之下，今天收回下軌之上"),
}
# 賣點：多頭轉弱或短線過熱的型態
SELL_SIGNALS = {
    "break_ma60": ("跌破季線", "收盤由季線之上跌到季線之下（中期趨勢轉弱）"),
    "break_ma240": ("跌破年線", "收盤由年線之上跌到年線之下（長期趨勢轉弱）"),
    "rsi_fall": ("RSI 超買回落", "RSI(14) 從 70 以上掉回 70 以下"),
    "kd_cross_high": ("KD 高檔死亡交叉", "前一天 K 值在 75 以上，今天由上往下穿過 D 值"),
    "macd_fade": ("MACD 紅柱縮短", "MACD 柱狀體仍為正，但連續 3 天縮短（漲勢減弱）"),
    "bb_fall": ("布林上緣回落", "前一天收在布林上軌之上，今天收回上軌之下"),
}
# 均線交叉：短天期均線穿過長天期均線，代表這段漲勢／跌勢開始轉向。
# 只拿來顯示與統計，不算進上面的「底部訊號」與趨勢分數（買賣規則用的是下面的均線排列）。
MA_BUY = {
    "ma5x10_up": ("5 日線上穿 10 日線", "5 日均線由下往上穿過 10 日均線（短線轉強）"),
    "ma10x20_up": ("10 日線上穿月線", "10 日均線由下往上穿過 20 日均線（短中期轉強）"),
    "ma20x60_up": ("月線上穿季線", "20 日均線由下往上穿過 60 日均線（中期轉強，黃金交叉）"),
}
MA_SELL = {
    "ma5x10_down": ("5 日線跌破 10 日線", "5 日均線由上往下穿過 10 日均線（短線轉弱）"),
    "ma10x20_down": ("10 日線跌破月線", "10 日均線由上往下穿過 20 日均線（短中期轉弱）"),
    "ma20x60_down": ("月線跌破季線", "20 日均線由上往下穿過 60 日均線（中期轉弱，死亡交叉）"),
}
MA_CROSSES = {"ma5x10": ("ma5", "ma10"), "ma10x20": ("ma10", "ma20"), "ma20x60": ("ma20", "ma60")}
ALL_SIGNALS = {**SIGNALS, **SELL_SIGNALS, **MA_BUY, **MA_SELL}
# 均線排列：10 日線在月線之上＝多頭（上漲還沒結束）；10 日線在月線之下＝空頭（下跌還沒結束）。
# 不用 5 日線：5 日和 10 日太接近，交叉太頻繁。
ALIGNMENTS = {"bull": "多頭（10 日線在月線之上）：上漲還沒結束",
              "bear": "空頭（10 日線在月線之下）：下跌還沒結束",
              "mixed": "10 日線與月線糾結：方向還不明確",
              "unknown": "資料不足"}
HORIZONS = (20, 60)


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """輸入 add_all() 後的日線，回傳每天每個訊號是否成立（True/False）。"""
    out = pd.DataFrame(index=df.index)
    uptrend = (df["close"] > df["ma240"]) & (df["ma60"] > df["ma120"])
    out["pullback_ma60"] = uptrend & (df["low"] <= df["ma60"] * 1.01) & (df["close"] > df["ma60"])
    out["rsi_rebound"] = (df["rsi"].shift(1) < 30) & (df["rsi"] >= 30)
    out["kd_cross_low"] = (df["k"].shift(1) <= df["d"].shift(1)) & (df["k"] > df["d"]) & (df["k"] < 25)
    h = df["hist"]
    out["macd_turn"] = (h < 0) & (h > h.shift(1)) & (h.shift(1) > h.shift(2)) & (h.shift(2) > h.shift(3))
    out["bb_rebound"] = (df["close"].shift(1) < df["bb_low"].shift(1)) & (df["close"] > df["bb_low"])
    c, pc = df["close"], df["close"].shift(1)
    out["break_ma60"] = (pc >= df["ma60"].shift(1)) & (c < df["ma60"])
    out["break_ma240"] = (pc >= df["ma240"].shift(1)) & (c < df["ma240"])
    out["rsi_fall"] = (df["rsi"].shift(1) > 70) & (df["rsi"] <= 70)
    out["kd_cross_high"] = (df["k"].shift(1) >= df["d"].shift(1)) & (df["k"] < df["d"]) & (df["k"].shift(1) > 75)
    out["macd_fade"] = (h > 0) & (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3))
    out["bb_fall"] = (pc > df["bb_up"].shift(1)) & (c < df["bb_up"])
    for key, (fast, slow) in MA_CROSSES.items():
        above, was = df[fast] > df[slow], df[fast].shift(1) > df[slow].shift(1)
        known = df[[fast, slow]].notna().all(axis=1) & df[[fast, slow]].shift(1).notna().all(axis=1)
        out[key + "_up"] = known & above & ~was
        out[key + "_down"] = known & ~above & was
    return out.fillna(False).astype(bool)


def ma_alignment(df: pd.DataFrame) -> pd.Series:
    """每天的均線排列：bull / bear / mixed / unknown。"""
    up = df["ma10"] > df["ma20"]
    down = df["ma10"] < df["ma20"]
    known = df[["ma10", "ma20"]].notna().all(axis=1)
    return pd.Series(np.where(~known, "unknown", np.where(up, "bull", np.where(down, "bear", "mixed"))), index=df.index)


def buy_score(sig: pd.DataFrame, window: int = 5) -> pd.Series:
    """最近 window 個交易日內出現過幾種不同的買點訊號（0～5）。"""
    return sig[list(SIGNALS)].rolling(window, min_periods=1).max().sum(axis=1).astype(int)


def sell_score(sig: pd.DataFrame, window: int = 5) -> pd.Series:
    """最近 window 個交易日內出現過幾種不同的賣點訊號（0～6）。"""
    return sig[list(SELL_SIGNALS)].rolling(window, min_periods=1).max().sum(axis=1).astype(int)


# ---------- 趨勢強弱：把多個條件合成「強多／偏多／中性／偏空／強空」 ----------
# 這是「現在的狀態」，會落後轉折；要不要買賣看下面的 simulate()（明確的進出場規則）。
# 狀態條件：成立 +1、不成立 -1、資料不足 0
CHECKS = {
    "above_ma20": ("收盤在月線之上", lambda d: d["close"] > d["ma20"], ["ma20"]),
    "above_ma60": ("收盤在季線之上", lambda d: d["close"] > d["ma60"], ["ma60"]),
    "ma20_up": ("月線上揚（高於 5 天前）", lambda d: d["ma20"] > d["ma20"].shift(5), ["ma20"]),
    "macd_pos": ("MACD 柱狀體為正（DIF > DEA）", lambda d: d["hist"] > 0, ["hist"]),
    "kd_up": ("K 值在 D 值之上", lambda d: d["k"] > d["d"], ["k", "d"]),
    "rsi_strong": ("RSI(14) 在 50 之上", lambda d: d["rsi"] > 50, ["rsi"]),
}
VERDICTS = {2: "強多", 1: "偏多", 0: "中性", -1: "偏空", -2: "強空"}
STRONG, LEAN = 5, 2          # 總分 ≥ 5 強多、≥ 2 偏多；≤ -5 強空、≤ -2 偏空
SIGNAL_CAP = 2               # 近 5 日買點／賣點訊號最多各算 2 分
OVERHEAT_RSI, OVERSOLD_RSI, OVERHEAT_RET5 = 80, 20, 0.25


def check_table(df: pd.DataFrame) -> pd.DataFrame:
    """每天每個狀態條件的分數：+1 / -1 / 0（資料不足）。"""
    out = pd.DataFrame(index=df.index)
    for key, (_, fn, need) in CHECKS.items():
        ok = fn(df)
        known = df[need].notna().all(axis=1)
        if key == "ma20_up":
            known &= df["ma20"].shift(5).notna()
        out[key] = np.where(known, np.where(ok, 1, -1), 0)
    return out


def verdict(df: pd.DataFrame, sig: pd.DataFrame, overheat_rsi: float = OVERHEAT_RSI,
            oversold_rsi: float = OVERSOLD_RSI, overheat_ret5: float = OVERHEAT_RET5) -> pd.DataFrame:
    """每天的綜合判斷。

    總分 = 6 個狀態條件（各 ±1）＋ 近 5 日買點訊號種數（最多 2）－ 近 5 日賣點訊號種數（最多 2），範圍 -8～+8。
    等級：≥5 強多、2～4 偏多、-1～1 中性、-4～-2 偏空、≤-5 強空。
    過熱（RSI ≥ 80 或 5 日漲幅 ≥ 25%）時不給強多／偏多，最高只到中性；
    超賣（RSI ≤ 20）時不給強空／偏空，最低只到中性。
    """
    checks = check_table(df)
    buys = buy_score(sig).clip(upper=SIGNAL_CAP)
    sells = sell_score(sig).clip(upper=SIGNAL_CAP)
    total = checks.sum(axis=1) + buys - sells
    level = pd.Series(np.select([total >= STRONG, total >= LEAN, total <= -STRONG, total <= -LEAN],
                                [2, 1, -2, -1], 0), index=df.index)
    hot = (df["rsi"] >= overheat_rsi) | (df["close"] / df["close"].shift(5) - 1 >= overheat_ret5)
    cold = df["rsi"] <= oversold_rsi
    capped = pd.Series("", index=df.index)
    capped[hot & (level > 0)] = "過熱"
    capped[cold & (level < 0)] = "超賣"
    level = level.where(~(hot & (level > 0)), 0).where(~(cold & (level < 0)), 0)
    return pd.DataFrame({"total": total.astype(int), "buys": buys, "sells": sells,
                         "level": level.astype(int), "capped": capped}).join(checks)


# ---------- 牛市／熊市／盤整 ----------
# 用年線（240 日均線）判斷：收盤在年線之上且年線比 20 天前高 → 牛市；
# 收盤在年線之下且年線比 20 天前低 → 熊市；其他 → 盤整；資料不滿 260 天 → 資料不足。
# 年線反應很慢，轉換期一定會誤判；它只用來決定「用哪一套買賣方法」，不單獨當買賣訊號。
REGIMES = {"bull": "牛市", "bear": "熊市", "range": "盤整", "unknown": "資料不足"}
REGIME_SLOPE_DAYS = 20


def regime(df: pd.DataFrame) -> pd.Series:
    ma = df["ma240"]
    slope = ma - ma.shift(REGIME_SLOPE_DAYS)
    out = np.where(ma.isna() | slope.isna(), "unknown",
                   np.where((df["close"] > ma) & (slope > 0), "bull",
                            np.where((df["close"] < ma) & (slope < 0), "bear", "range")))
    return pd.Series(out, index=df.index)


# ---------- 明確的進出場規則：什麼時候買、什麼時候賣 ----------
# 趨勢分數用「均線、MACD」等狀態，等它轉強時通常已經漲一段、轉弱時已經跌一段，
# 照它買賣常常買在半山腰、賣在山腳。所以買賣改用下面的規則。
#
# 兩套做法（每次買進都投入相同金額，稱為 1 份）：
#   all_in（全進全出）：
#     買進：底部確認（近 5 日出現過任一種買點 ▲，且今天 K > D、收盤站上 5 日線）
#     賣出全部：KD 高檔死叉或 RSI 超買回落（獲利了結）、收盤跌破 10 日線、收盤比成本低 8%（停損）
#   scale（低點加倉、高點賣一半）：
#     買進／加碼 1 份：底部確認「剛成立」那天，或多頭回檔再轉強（持有加上這 1 份不超過 3 份）
#     賣出一半：KD 高檔死叉或 RSI 超買回落（區域高點）；持有份數跟著減半（3 份 → 1.5 份）
#     賣出全部：收盤跌破 10 日線、收盤比平均成本低 8%（停損）
#   auto（依牛熊切換）：牛市用 scale，熊市、盤整、資料不足用 all_in。
#   winrate（高勝率，預設）：不使用上面任何條件，另外用 2023–2025 的歷史資料找出來的規則——
#     買進：收盤在季線（60 日均線）之上，且 RSI(3) < 25（多頭裡的短線急跌）
#     賣出：RSI(3) > 70（反彈到短線過熱），或持有滿 20 個交易日（時間停損）
#     在 54 檔權值股上：2023–24 勝率 65%，沒參與挑選的 2025 年 70%、2026 年 81%（扣交易成本後）。
#     不設價格停損：測試過加停損反而讓勝率下降（停損 8% 時 2026 年只剩 53%）。
# 訊號用收盤價判斷，隔天開盤價成交（收盤後才知道訊號，實際只能隔天買賣）。
# 空手時用底部確認買進，底部訊號必須是上次賣出之後才出現的，避免賣掉隔天又用同一個舊訊號買回。
STOP_LOSS = 0.08
MAX_UNITS = 3
MA10_TURN_DAYS = 3        # 10 日線比幾天前高，算「止跌」
FEE = 0.001425             # 券商手續費（未打折），買、賣各收一次
TAX_STOCK, TAX_ETF = 0.003, 0.001   # 證券交易稅，賣出時收
WIN_TREND = "ma60"         # 高勝率模式：收盤要在季線之上
WIN_BUY_BELOW = 25         # RSI(3) 低於此值買進
WIN_SELL_ABOVE = 70        # RSI(3) 高於此值賣出
WIN_MAX_DAYS = 20          # 最多持有幾個交易日


def tax_for(code: str) -> float:
    """ETF（代號 00 開頭）證交稅 0.1%，股票 0.3%。"""
    return TAX_ETF if str(code).startswith("00") else TAX_STOCK


@dataclass(frozen=True)
class TradeSettings:
    """買賣規則的設定（對應 config.yaml 的 trading 區塊）。"""
    mode: str = "auto"              # all_in / scale / auto
    regime_from: str = "0050"       # 牛熊用哪一檔的年線；"self" = 每檔用自己的
    max_units: float = MAX_UNITS
    stop: float = STOP_LOSS
    ma_entry: bool = True           # 買進／加碼要 10 日線止跌
    ma_hold: bool = False           # 10 日線在月線之上時，區域高點不賣
    ma10_turn_days: int = MA10_TURN_DAYS
    fee: float = FEE                # 手續費（買賣各一次）
    # 高勝率模式的參數（每支股票可以不同，見 optimize.py）
    win_trend: str = "ma60"         # 收盤要在哪條均線之上（"none" = 不看）
    win_osc: str = "rsi3"           # 用 rsi2 或 rsi3
    win_buy_below: float = 25       # 低於此值買進
    win_sell_above: float = 70      # 高於此值賣出
    win_max_days: int = 20          # 最多持有幾個交易日


def mode_for(mode: str, regime: str) -> str:
    """auto 模式依牛熊決定當天用哪一套：牛市 scale，其他 all_in。"""
    if mode != "auto":
        return mode  # all_in / scale / winrate 都不看牛熊
    return "scale" if regime == "bull" else "all_in"


MODES = {"winrate": "高勝率（季線之上 RSI 短線超賣）", "all_in": "全進全出", "scale": "低點加倉、高點賣一半",
         "auto": "依牛熊切換"}
TREND_NAMES = {"none": "不看均線", "ma20": "月線", "ma60": "季線", "ma120": "半年線", "ma240": "年線"}


def winrate_rule(st: "TradeSettings") -> str:
    """把高勝率模式的參數寫成一句中文規則。"""
    osc = {"rsi2": "RSI(2)", "rsi3": "RSI(3)"}[st.win_osc]
    trend = "" if st.win_trend == "none" else f"收盤在{TREND_NAMES[st.win_trend]}之上，而且 "
    return (f"買進：{trend}{osc} 低於 {st.win_buy_below:g}；"
            f"賣出：{osc} 高於 {st.win_sell_above:g}，或持有滿 {st.win_max_days} 個交易日")


WINRATE_RULE = winrate_rule(TradeSettings())
ENTRY_RULE = "近 5 日出現過任一種買點 ▲，而且今天 K 值在 D 值之上、收盤站上 5 日線（賣出之後要出現新的底部訊號才會再買）"
PULLBACK_RULE = "多頭回檔再轉強：收盤在月線之上、月線往上且在季線之上，近 5 日 K 值曾低於 50，今天 K 值上穿 D 值且收盤站上 5 日線"
REASONS = {
    "bottom": "底部確認",
    "pullback": "多頭回檔再轉強",
    "take_profit": "KD 高檔死叉或 RSI 超買回落（區域高點）",
    "below_ma10": "收盤跌破 10 日線（短線轉弱）",
    "stop_loss": "收盤比平均成本低 {stop:.0%}（停損）",
    "rsi3_low": f"收盤在季線之上，RSI(3) 低於 {WIN_BUY_BELOW}（多頭裡的短線急跌）",
    "rsi3_high": f"RSI(3) 高於 {WIN_SELL_ABOVE}（反彈到短線過熱）",
    "max_days": f"持有滿 {WIN_MAX_DAYS} 個交易日（時間停損）",
}
ACTIONS = {"buy": "買進", "add": "加碼", "half": "賣出一半", "exit": "全部賣出"}


def entry_signal(df: pd.DataFrame, sig: pd.DataFrame) -> pd.Series:
    """底部確認：近 5 日出現過底部訊號，且今天 K > D、收盤站上 5 日線。"""
    recent_bottom = sig[list(SIGNALS)].rolling(5, min_periods=1).max().astype(bool).any(axis=1)
    return (recent_bottom & (df["k"] > df["d"]) & (df["close"] > df["ma5"])).fillna(False)


def pullback_signal(df: pd.DataFrame) -> pd.Series:
    """多頭回檔再轉強：多頭排列中 K 值回檔到 50 以下後，今天重新上穿 D 值。"""
    k_cross = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1))
    dipped = df["k"].rolling(5, min_periods=1).min() < 50
    uptrend = (df["close"] > df["ma20"]) & (df["ma20"] > df["ma20"].shift(5)) & (df["ma20"] > df["ma60"])
    return (uptrend & k_cross & dipped & (df["close"] > df["ma5"])).fillna(False)


def simulate(df: pd.DataFrame, sig: pd.DataFrame, mode: str = "all_in", regimes: pd.Series | None = None,
             settings: TradeSettings = TradeSettings(), tax: float = 0.0) -> dict:
    """依規則逐日模擬。

    mode：all_in / scale / auto（auto 需要 regimes：每天的 bull/bear/range/unknown）。
          不看 settings.mode，方便同一份設定拿來比較三種做法。
    settings.ma_entry：買進／加碼還要「10 日線止跌」（10 日線比 ma10_turn_days 天前高，下跌告一段落才買）。
              不用「10 日線在月線之上」：底部訊號出現時 10 日線幾乎都還在月線之下，等它站上月線已經漲一大段，
              回測只剩 37 筆、賺錢比例 27%。
    settings.ma_hold：10 日線在月線之上時，不因「區域高點」賣出（上漲還沒結束就續抱）；
             跌破 10 日線、停損照常賣。10 日線一跌破月線，區域高點就恢復照賣。
    回傳：
      rounds：每一輪（從空手買進到全部賣出）{start_i, end_i, actions, cost, pnl, ret, max_units}；
              還沒賣完的那一輪 end_i = None，損益用最新收盤計算。
      actions 每一筆 {i（成交日＝訊號隔天）, type: buy/add/half/exit, px, reason, units（成交後持有份數，賣一半會減半）, mode}
      pending：最後一天收盤出現、要在下一個交易日開盤執行的動作 {type, reason, mode} 或 None。
      position：目前持有 {units（份數，可能是 1.5 這種小數）, avg_cost, ret} 或 None。
    金額單位都是「份」：每次買進投入 1 份資金。
    交易成本：買進付 settings.fee、賣出付 settings.fee + tax（證交稅，用 tax_for(代號) 算），報酬都是扣完成本的。
    """
    if mode == "auto" and regimes is None:
        raise ValueError("auto 模式需要 regimes")
    n = len(df)
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    ma10 = df["ma10"].to_numpy()
    st = settings
    short_up = ((df["ma10"] > df["ma10"].shift(st.ma10_turn_days)).to_numpy() if st.ma_entry
                else np.ones(n, bool))
    bottom = entry_signal(df, sig).to_numpy() & short_up
    bottom_new = bottom & ~np.r_[False, bottom[:-1]]
    pull = pullback_signal(df).to_numpy() & short_up
    top = (sig["kd_cross_high"] | sig["rsi_fall"]).to_numpy()
    if st.ma_hold:
        top = top & (ma_alignment(df) != "bull").to_numpy()
    reg = regimes.to_numpy() if regimes is not None else None
    osc = df[st.win_osc]
    above = (df["close"] > df[st.win_trend]) if st.win_trend != "none" else pd.Series(True, index=df.index)
    win_buy = (above & (osc < st.win_buy_below)).fillna(False).to_numpy()
    osc = osc.to_numpy()
    any_bottom = sig[list(SIGNALS)].any(axis=1).to_numpy()
    last_bottom = pd.Series(np.where(any_bottom, np.arange(n), np.nan)).ffill().fillna(-1).to_numpy()

    rounds, pending = [], None
    shares = cost = 0.0          # cost＝目前投入的份數（賣一半就減半）
    last_exit = -1               # 上次出清訊號那天
    rnd = None
    buy_cost, sell_keep = 1 + st.fee, 1 - st.fee - tax
    for t in range(n):
        m = mode_for(mode, reg[t] if reg is not None else "unknown")
        act = reason = None
        if m == "winrate":
            if shares == 0:
                if win_buy[t]:
                    act, reason = "buy", "rsi3_low"
            elif osc[t] > st.win_sell_above:
                act, reason = "exit", "rsi3_high"
            elif t - rnd["start_i"] + 1 >= st.win_max_days:
                act, reason = "exit", "max_days"
        elif shares == 0:
            if bottom[t] and last_bottom[t] > last_exit:
                act, reason = "buy", "bottom"
            elif m == "scale" and pull[t]:
                act, reason = "buy", "pullback"
        else:
            below10 = not np.isnan(ma10[t]) and c[t] < ma10[t]
            if c[t] <= cost / shares * (1 - st.stop):
                act, reason = "exit", "stop_loss"
            elif m == "all_in":
                if top[t]:
                    act, reason = "exit", "take_profit"
                elif below10:
                    act, reason = "exit", "below_ma10"
            else:
                if below10:
                    act, reason = "exit", "below_ma10"
                elif top[t]:
                    act, reason = "half", "take_profit"
                elif cost + 1 <= st.max_units + 1e-9 and (bottom_new[t] or pull[t]):
                    act, reason = "add", "bottom" if bottom_new[t] else "pullback"
        if act is None:
            continue
        if t + 1 >= n:
            pending = {"type": act, "reason": reason, "mode": m}
            continue
        px = float(o[t + 1])
        if act in ("buy", "add"):
            if act == "buy":
                rnd = {"start_i": t + 1, "end_i": None, "actions": [], "cost": 0.0, "cash": 0.0, "max_units": 0}
                rounds.append(rnd)
            shares += 1 / (px * buy_cost)
            cost += 1.0
            rnd["cost"] += 1.0
            rnd["max_units"] = max(rnd["max_units"], cost)
        elif act == "half":
            rnd["cash"] += shares / 2 * px * sell_keep
            shares /= 2
            cost /= 2
        else:
            rnd["cash"] += shares * px * sell_keep
            shares = cost = 0.0
            last_exit = t
            rnd["end_i"] = t + 1
        rnd["actions"].append({"i": t + 1, "type": act, "px": px, "reason": reason, "units": round(cost, 4), "mode": m})

    for r in rounds:
        held = shares if r["end_i"] is None else 0.0
        r["pnl"] = r.pop("cash") + held * float(c[-1]) * sell_keep - r["cost"]
        r["ret"] = r["pnl"] / r["cost"]
    position = None
    if shares > 0:
        position = {"units": round(cost, 4), "avg_cost": cost / shares, "ret": float(c[-1]) * shares * sell_keep / cost - 1,
                    "start_i": rnd["start_i"], "days": n - rnd["start_i"]}
    return {"rounds": rounds, "pending": pending, "position": position}


def merge_sims(parts: list[tuple[dict, int]]) -> dict:
    """把好幾段資料各自模擬的結果接起來：parts = [(simulate 結果, 這段在整體的起始位置)]。

    每段的日期位置加上 offset；pending、position 只看最後一段（最新的狀態）。
    """
    rounds = []
    for sim, off in parts:
        for r in sim["rounds"]:
            r = dict(r, start_i=r["start_i"] + off, end_i=None if r["end_i"] is None else r["end_i"] + off,
                     actions=[dict(a, i=a["i"] + off) for a in r["actions"]])
            rounds.append(r)
    last, off = parts[-1] if parts else ({"pending": None, "position": None}, 0)
    pos = last["position"] and dict(last["position"], start_i=last["position"]["start_i"] + off)
    return {"rounds": rounds, "pending": last["pending"], "position": pos}


def round_stats(rounds: list[dict]) -> dict:
    """已完成的輪次：輪數、賺錢比例、平均每輪報酬；總損益（份）與報酬率（總損益 ÷ 最多同時投入的份數）。"""
    closed = [r for r in rounds if r["end_i"] is not None]
    if not closed:
        return {"n": 0, "win": None, "avg": None, "pnl": None, "roc": None, "max_units": 0}
    rets = np.array([r["ret"] for r in closed])
    pnl = float(sum(r["pnl"] for r in closed))
    mu = max(r["max_units"] for r in closed)
    return {"n": int(len(closed)), "win": float((rets > 0).mean()), "avg": float(rets.mean()),
            "pnl": pnl, "roc": pnl / mu, "max_units": float(mu)}


def pool_round_stats(per_stock: list[dict]) -> dict:
    """多檔合計：輪次加總；報酬率取各檔報酬率的平均。"""
    have = [s for s in per_stock if s["n"]]
    if not have:
        return {"n": 0, "win": None, "avg": None, "pnl": None, "roc": None, "stocks": 0}
    n = sum(s["n"] for s in have)
    return {"n": n, "win": sum(s["win"] * s["n"] for s in have) / n, "avg": sum(s["avg"] * s["n"] for s in have) / n,
            "pnl": sum(s["pnl"] for s in have), "roc": float(np.mean([s["roc"] for s in have])), "stocks": len(have)}


def forward_returns(close: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    return pd.DataFrame({f"ret{h}": close.shift(-h) / close - 1 for h in horizons})


def backtest(df: pd.DataFrame, sig: pd.DataFrame, horizons=HORIZONS, extra: dict | None = None) -> dict:
    """每個訊號：出現次數、之後 N 日平均報酬、上漲比例；同時給「任意一天買進」當比較基準。
    extra：額外要回測的 {名稱: 每天 True/False}，例如綜合判斷的轉折日。"""
    fwd = forward_returns(df["close"], horizons)
    masks = {k: sig[k] for k in ALL_SIGNALS}
    masks.update(extra or {})
    masks["_any_day"] = pd.Series(True, index=df.index)
    result = {}
    for name, mask in masks.items():
        mask = mask.fillna(False).astype(bool)
        stats = {"count": int(mask.sum())}
        for h in horizons:
            r = fwd.loc[mask, f"ret{h}"].dropna()
            stats[f"n{h}"] = int(len(r))
            stats[f"avg{h}"] = float(r.mean()) if len(r) else None
            stats[f"win{h}"] = float((r > 0).mean()) if len(r) else None
        result[name] = stats
    return result


def pool_backtests(per_stock: list[dict]) -> dict:
    """把多檔股票的回測加總（依樣本數加權），得到訊號在整個股票池的表現。"""
    pooled = {}
    for name in (per_stock[0] if per_stock else {}):
        agg = {"count": sum(b[name]["count"] for b in per_stock)}
        for h in HORIZONS:
            n = sum(b[name][f"n{h}"] for b in per_stock)
            agg[f"n{h}"] = n
            for key in ("avg", "win"):
                agg[f"{key}{h}"] = (sum(b[name][f"{key}{h}"] * b[name][f"n{h}"]
                                        for b in per_stock if b[name][f"n{h}"]) / n) if n else None
        pooled[name] = agg
    return pooled
