"""盤中即時報價（證交所「基本市況報導」公開 API），以及「如果今天收在現價」的規則試算。

    quotes = fetch_quotes({"0050": "TWSE", "6488": "TPEx"})
    check = intraday_check(daily_history, quotes.loc["0050"], settings, tax)

資料約每 5 秒更新一次；桌面程式每分鐘查一次，一次最多 BATCH 檔，避免被暫時封鎖。
盤中的開高低收、成交量都還會變，試算結果只代表「如果收盤價就是現在這個價格」。
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

from . import indicators as ind
from . import signals as sg

URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
BATCH = 50
MIN_INTERVAL = 2.0     # 兩次請求之間至少隔幾秒
TIMEOUT = 15
EX = {"TWSE": "tse", "TPEx": "otc"}
COLUMNS = ["name", "market", "date", "time", "price", "open", "high", "low", "prev_close", "volume",
           "bid", "ask", "limit_up", "limit_down", "change", "change_pct"]

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (tw-stock-screener desktop)", "Accept": "application/json"})
_last_call = 0.0


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if x > 0 else float("nan")


def _first(s: str | None) -> float:
    """五檔報價字串 "97.1000_97.0500_..." 取第一檔。"""
    return _num((s or "").split("_")[0])


def parse(payload: dict) -> pd.DataFrame:
    """把 API 回傳的 JSON 轉成以代號為索引的表格。

    成交價 z 在兩筆成交之間會是 "-"，這時依序改用 pz（上一筆成交價）、買一與賣一的中間價。
    """
    rows = {}
    for m in (payload or {}).get("msgArray", []):
        price = _num(m.get("z"))
        if np.isnan(price):
            price = _num(m.get("pz"))
        bid, ask = _first(m.get("b")), _first(m.get("a"))
        if np.isnan(price) and not (np.isnan(bid) or np.isnan(ask)):
            price = (bid + ask) / 2
        prev = _num(m.get("y"))
        date = pd.to_datetime(m.get("d"), format="%Y%m%d", errors="coerce")
        rows[str(m.get("c"))] = {
            "name": m.get("n", ""), "market": {"tse": "TWSE", "otc": "TPEx"}.get(m.get("ex"), m.get("ex")),
            "date": date, "time": m.get("t", ""), "price": price,
            "open": _num(m.get("o")), "high": _num(m.get("h")), "low": _num(m.get("l")), "prev_close": prev,
            "volume": _num(m.get("v")) * 1000,   # API 單位是張，轉成股（跟日資料一致）
            "bid": bid, "ask": ask, "limit_up": _num(m.get("u")), "limit_down": _num(m.get("w")),
            "change": price - prev, "change_pct": price / prev - 1,
        }
    df = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
    df.index.name = "code"
    return df


def fetch_quotes(codes: dict[str, str]) -> pd.DataFrame:
    """codes = {代號: "TWSE" 或 "TPEx"}。每 BATCH 檔一次請求，失敗的那批略過並記在 df.attrs["errors"]。"""
    global _last_call
    items = [f"{EX.get(m, 'tse')}_{c}.tw" for c, m in codes.items()]
    frames, errors = [], []
    for i in range(0, len(items), BATCH):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            r = _session.get(URL, params={"ex_ch": "|".join(items[i:i + BATCH]), "json": 1, "delay": 0},
                             timeout=TIMEOUT)
            r.raise_for_status()
            frames.append(parse(r.json()))
        except (requests.RequestException, ValueError) as exc:
            errors.append(str(exc))
    out = pd.concat(frames) if frames else parse({})
    out.attrs["errors"] = errors
    return out


def provisional_daily(daily: pd.DataFrame, q: pd.Series) -> pd.DataFrame:
    """把盤中報價當成「今天的日 K」接到歷史後面（收盤價＝現價）。今天已經有收盤資料時直接回傳原資料。"""
    if pd.isna(q.get("price")) or pd.isna(q.get("date")):
        return daily
    day = pd.Timestamp(q["date"]).normalize()
    if len(daily) and daily["date"].iat[-1] >= day:
        return daily
    o = q["open"] if pd.notna(q["open"]) else q["price"]
    bar = {"date": day, "open": o, "high": np.nanmax([q["high"], q["price"]]), "low": np.nanmin([q["low"], q["price"]]),
           "close": q["price"], "volume": q["volume"] if pd.notna(q["volume"]) else 0.0}
    return pd.concat([daily, pd.DataFrame([bar])], ignore_index=True)


def rsi_cross_price(closes: pd.Series, target: float, n: int = 3) -> float | None:
    """今天收在多少，RSI(n) 會剛好等於 target。RSI 隨收盤價單調遞增，用二分法求解。"""
    closes = closes.dropna().reset_index(drop=True)
    if len(closes) <= n + 1:
        return None
    last = float(closes.iat[-1])

    def rsi_at(p: float) -> float:
        return float(ind.rsi(pd.concat([closes, pd.Series([p])], ignore_index=True), n).iat[-1])

    lo, hi = last * 0.5, last * 1.5
    if not (rsi_at(lo) < target < rsi_at(hi)):
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if rsi_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def intraday_check(daily: pd.DataFrame, q: pd.Series, st: sg.TradeSettings = sg.TradeSettings(),
                   tax: float = sg.TAX_STOCK) -> dict:
    """如果今天收盤價就是現價，高勝率規則會怎麼判斷？

    daily：到昨天為止的日 K（date, open, high, low, close, volume，已還原拆股）。
    回傳 {status, action, price, rsi3, ma60, buy_below, sell_above, days_held, lines}：
      status：flat（空手）／hold（持有中）
      action：buy／sell／None——收在現價時，收盤後會出現的訊號（隔天開盤執行）
      buy_below：空手時，收盤低於這個價格 RSI(3) 就會低於買進門檻
      sell_above：持有時，收盤高於這個價格 RSI(3) 就會高於賣出門檻
    """
    hist = daily.reset_index(drop=True)
    if len(hist):
        h = ind.add_all(hist)
        base = sg.simulate(h, sg.detect(h), "winrate", settings=st, tax=tax)
    else:
        base = {"position": None, "pending": None}
    full = provisional_daily(hist, q)
    d = ind.add_all(full)
    sim = sg.simulate(d, sg.detect(d), "winrate", settings=st, tax=tax)
    last = d.iloc[-1]
    price = float(last["close"])
    osc_now = float(last[st.win_osc]) if pd.notna(last[st.win_osc]) else float("nan")
    trend = float(last[st.win_trend]) if st.win_trend != "none" and pd.notna(last[st.win_trend]) else float("nan")
    n = {"rsi2": 2, "rsi3": 3}[st.win_osc]
    pos = sim["position"]
    # 昨天收盤後已經有待執行的動作（今天開盤就該做），優先提醒
    carry = base.get("pending")
    out = {"price": price, "osc": osc_now, "trend": trend, "status": "hold" if pos else "flat",
           "action": None, "buy_below": None, "sell_above": None, "days_held": pos["days"] if pos else 0,
           "carry": carry["type"] if carry else None, "lines": []}
    osc_name = {"rsi2": "RSI(2)", "rsi3": "RSI(3)"}[st.win_osc]
    trend_name = sg.TREND_NAMES.get(st.win_trend, st.win_trend)
    if carry:
        out["lines"].append(f"⚠ 昨天收盤已出現{'買進' if carry['type'] == 'buy' else '賣出'}訊號，今天開盤就該執行。")
    if sim["pending"]:
        out["action"] = "buy" if sim["pending"]["type"] in ("buy", "add") else "sell"
    if pos:
        out["sell_above"] = rsi_cross_price(full["close"].iloc[:-1], st.win_sell_above, n)
        left = st.win_max_days - pos["days"]
        out["lines"].append(f"持有中第 {pos['days']} 天，以現價計算 {pos['ret']:+.1%}（已扣成本）。")
        if out["sell_above"]:
            out["lines"].append(f"收盤高於 {out['sell_above']:.2f}，{osc_name} 就會超過 {st.win_sell_above:g} → 收盤後出現賣出訊號。")
        out["lines"].append(f"再 {max(left, 0)} 個交易日滿 {st.win_max_days} 天，不論賺賠都賣出。")
    else:
        out["buy_below"] = rsi_cross_price(full["close"].iloc[:-1], st.win_buy_below, n)
        if st.win_trend != "none":
            ok = not np.isnan(trend) and price > trend
            out["lines"].append(f"{'✓' if ok else '✗'} 現價 {price:.2f} {'高於' if ok else '不高於'}{trend_name} {trend:.2f}")
        if out["buy_below"]:
            out["lines"].append(f"收盤低於 {out['buy_below']:.2f}，{osc_name} 就會低於 {st.win_buy_below:g}（目前 {osc_now:.0f}）。")
    if out["action"] == "buy":
        out["lines"].insert(0, "▲ 如果收在現價：收盤後出現買進訊號（下一個交易日開盤買）。")
    elif out["action"] == "sell":
        out["lines"].insert(0, "▼ 如果收在現價：收盤後出現賣出訊號（下一個交易日開盤賣）。")
    else:
        out["lines"].insert(0, "如果收在現價：不會出現新的買賣訊號。")
    return out


def market_open(now: dt.datetime | None = None) -> bool:
    """台股盤中（週一到週五 9:00–13:30，台北時間；不判斷國定假日）。"""
    now = now or dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    return now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(13, 30)
