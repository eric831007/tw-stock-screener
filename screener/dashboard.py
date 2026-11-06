"""權值股技術分析儀表板：讀取已存的日資料，計算指標與買點訊號，產生一個獨立的 HTML 檔。

    python -m screener dashboard            # 產生 dashboard/index.html

HTML 不需要網路：圖表程式和資料都內嵌在檔案裡，點兩下就能用瀏覽器打開。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd
import yaml

from . import indicators as ind
from . import signals as sg
from . import store

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
OUT_FILE = ROOT / "dashboard" / "index.html"
FIELDS = ["open", "high", "low", "close", "volume", "rsi", "dif", "dea", "hist", "k", "d",
          "bb_up", "bb_mid", "bb_low"]
OHLCV = ["date", "open", "high", "low", "close", "volume"]


# ---------- 設定 ----------
def _config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}


def load_watchlist() -> dict:
    return _config().get("watchlist", {})


def load_overheat() -> dict:
    """config.yaml 的過熱／超賣門檻 → verdict() 的參數。"""
    oh = _config().get("overheat", {}) or {}
    return {"overheat_rsi": float(oh.get("rsi_above", sg.OVERHEAT_RSI)),
            "overheat_ret5": float(oh.get("return_5d_above_pct", sg.OVERHEAT_RET5 * 100)) / 100,
            "oversold_rsi": float(oh.get("oversold_rsi_below", sg.OVERSOLD_RSI))}


def load_trading() -> sg.TradeSettings:
    """config.yaml 的 trading 設定。"""
    tr = _config().get("trading", {}) or {}
    d = sg.TradeSettings()
    mode = str(tr.get("mode", d.mode))
    if mode not in sg.MODES:
        raise SystemExit(f"config.yaml trading.mode 只能是 {', '.join(sg.MODES)}，目前是 {mode}")
    return sg.TradeSettings(
        mode=mode, regime_from=str(tr.get("regime_from", d.regime_from)),
        max_units=float(tr.get("max_units", d.max_units)),
        stop=float(tr.get("stop_loss_pct", d.stop * 100)) / 100,
        ma_entry=bool(tr.get("ma_entry_filter", d.ma_entry)),
        ma_hold=bool(tr.get("ma_hold_in_uptrend", d.ma_hold)),
        ma10_turn_days=int(tr.get("ma10_turn_days", d.ma10_turn_days)))


# ---------- 資料 ----------
def daily_bars_raw(prices: pd.DataFrame, code: str) -> pd.DataFrame:
    """某一檔依日期排序的日 K（去掉沒有收盤價的日子），價格未還原。"""
    daily = prices[prices["code"] == code][OHLCV]
    return daily.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)


def daily_bars(prices: pd.DataFrame, code: str) -> pd.DataFrame:
    """同 daily_bars_raw，並還原拆股／合併造成的價格跳動。"""
    daily = daily_bars_raw(prices, code)
    return ind.adjust_splits(daily) if len(daily) > 1 else daily


def pick_watchlist(prices: pd.DataFrame, info: pd.DataFrame, cfg: dict) -> tuple[list[dict], str]:
    """回傳 [{code, name, group}], 以及排序依據說明。"""
    last_day = prices["date"].max()
    latest = prices[prices["date"] == last_day].set_index("code")
    stocks = latest[(latest["market"] == "TWSE") & latest.index.str.fullmatch(r"[1-9]\d{3}")]

    shares = info.set_index("code")["shares"] if "shares" in info else pd.Series(dtype=float)
    shares = pd.to_numeric(shares, errors="coerce")
    mcap = (stocks["close"] * shares.reindex(stocks.index)).dropna()
    top_n = int(cfg.get("top_market_cap", 50))
    if len(mcap) >= top_n:
        ranked, basis = mcap.sort_values(ascending=False).index[:top_n], "市值"
    else:  # 還沒有股數資料時，暫用近 60 日平均成交值排序
        recent = prices[(prices["market"] == "TWSE") & prices["code"].str.fullmatch(r"[1-9]\d{3}")]
        recent = recent[recent["date"] >= sorted(prices["date"].unique())[-60:][0]]
        ranked = recent.groupby("code")["value"].mean().sort_values(ascending=False).index[:top_n]
        basis = "近 60 日平均成交值（尚無股數資料，下次更新後改用市值）"

    names = latest["name"].to_dict()
    picked, seen = [], set()
    for group, codes in (("我的 ETF", cfg.get("my_etfs", [])), ("自選", cfg.get("extra", [])),
                         (f"市值前 {top_n}", list(ranked))):
        for code in map(str, codes):
            if code in seen or code not in names:
                continue
            seen.add(code)
            picked.append({"code": code, "name": names[code], "group": group})
    return picked, basis


def _series(df: pd.DataFrame) -> dict:
    cols = FIELDS + [c for c in df.columns if c.startswith("ma") and c[2:].isdigit()]
    out = {"t": [d.strftime("%Y-%m-%d") for d in df["date"]]}
    for c in cols:
        vals = df[c].to_numpy()
        digits = 0 if c == "volume" else 2
        out[c] = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), digits)
                  for v in vals]
    return out


# ---------- 畫面文字 ----------
def _verdict_now(v: pd.DataFrame, sig: pd.DataFrame) -> dict:
    """最新一天的趨勢強弱，以及每個條件是否成立，給畫面上的檢查表用。"""
    if v.empty:
        return {"level": 0, "label": sg.VERDICTS[0], "total": 0, "capped": "", "checks": [], "streak": 0,
                "buys": [], "sells": [], "buyPts": 0, "sellPts": 0}
    last = v.iloc[-1]
    lv = v["level"].to_numpy()
    streak = 1
    while streak < len(lv) and lv[-streak - 1] == lv[-1]:
        streak += 1
    recent = sig.iloc[-5:]
    return {
        "level": int(last["level"]), "label": sg.VERDICTS[int(last["level"])],
        "total": int(last["total"]), "capped": last["capped"], "streak": streak,
        "checks": [{"name": sg.CHECKS[k][0], "pt": int(last[k])} for k in sg.CHECKS],
        "buys": [sg.SIGNALS[k][0] for k in sg.SIGNALS if recent[k].any()],
        "sells": [sg.SELL_SIGNALS[k][0] for k in sg.SELL_SIGNALS if recent[k].any()],
        "buyPts": int(last["buys"]), "sellPts": int(last["sells"]),
    }


def _px(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _units(u: float) -> str:
    return f"{u:g}"


def _reason(code: str, stop: float) -> str:
    return sg.REASONS[code].format(stop=stop)


def _action(d: pd.DataFrame, sig: pd.DataFrame, sim: dict, mode_now: str, st: sg.TradeSettings,
            align: pd.Series | None = None) -> dict:
    """最新一天要做什麼：買進／加碼／賣出一半／全部賣出（下一個交易日開盤執行）、續抱、空手，並列出要盯的價位與條件。"""
    if d.empty:
        return {"code": "flat", "label": "空手", "lines": ["資料不足"]}
    day = lambda i: d["date"].iat[i].strftime("%m/%d")
    last = d.iloc[-1]
    pos, pend = sim["position"], sim["pending"]
    al = align.iat[-1] if align is not None and len(align) else "unknown"
    align_line = f"均線：{sg.ALIGNMENTS[al]}。"
    held = (f"持有 {_units(pos['units'])} 份，平均成本 {_px(pos['avg_cost'])}，目前 {pos['ret']:+.1%}" if pos else "")
    if pend:
        code = {"buy": "buy", "add": "add", "half": "half", "exit": "sell"}[pend["type"]]
        lines = [f"今天（{day(-1)}）收盤出現：{_reason(pend['reason'], st.stop)}。下一個交易日開盤{sg.ACTIONS[pend['type']]}。"]
        if pos:
            lines.insert(0, held + "。")
        if pend["type"] in ("buy", "add"):
            lines.append("每次買進投入相同金額（1 份）。")
        lines.append(align_line)
        return {"code": code, "label": sg.ACTIONS[pend["type"]], "lines": lines}
    if mode_now == "winrate":
        return _action_winrate(d, sim, held)
    if pos:
        lines = [held + "。"]
        if not pd.isna(last["ma10"]):
            lines.append(f"收盤跌破 10 日線（今天 {_px(last['ma10'])}）→ 全部賣出；停損價 {_px(pos['avg_cost'] * (1 - st.stop))}。")
        if st.ma_hold and al == "bull":
            lines.append("10 日線在月線之上：上漲還沒結束，區域高點先不賣；等 10 日線跌破月線後，區域高點才恢復照賣。")
        elif mode_now == "scale":
            lines.append("KD 高檔死叉或 RSI 從 70 以上回落 → 賣出一半。")
        else:
            lines.append("KD 高檔死叉或 RSI 從 70 以上回落 → 全部賣出（獲利了結）。")
        if mode_now == "scale" and pos["units"] + 1 <= st.max_units + 1e-9:
            lines.append(f"再出現底部確認或多頭回檔再轉強 → 加碼 1 份（目前 {_units(pos['units'])}／{_units(st.max_units)} 份）。")
        lines.append(align_line)
        return {"code": "hold", "label": "續抱", "lines": lines}
    recent_bottom = [sg.SIGNALS[k][0] for k in sg.SIGNALS if sig[k].iloc[-5:].any()]
    ok = lambda b: "✓" if b else "✗"
    lines = ["等底部確認的條件全部成立：",
             f"{ok(recent_bottom)} 近 5 日出現底部訊號" + (f"（{'、'.join(recent_bottom)}）" if recent_bottom else ""),
             f"{ok(last['k'] > last['d'])} K 值在 D 值之上（K {last['k']:.0f}／D {last['d']:.0f}）",
             f"{ok(last['close'] > last['ma5'])} 收盤站上 5 日線（{_px(last['ma5'])}）"]
    if st.ma_entry:
        n = st.ma10_turn_days
        prev10 = d["ma10"].iat[-1 - n] if len(d) > n else float("nan")
        if pd.isna(prev10) or pd.isna(last["ma10"]):
            lines.append(f"✗ 10 日線止跌（資料還不夠算 {n} 天前的 10 日線）")
        else:
            lines.append(f"{ok(last['ma10'] > prev10)} 10 日線止跌（今天 {_px(last['ma10'])}，{n} 天前 {_px(prev10)}）")
    if mode_now == "scale":
        lines.append("或：" + sg.PULLBACK_RULE)
    lines.append(align_line)
    done = [r for r in sim["rounds"] if r["end_i"] is not None]
    if done:
        r = done[-1]
        lines.append(f"上一輪：{day(r['start_i'])} 買進 → {day(r['end_i'])} 全部賣出（{r['ret']:+.1%}）")
        lines.append("（要等賣出之後出現新的底部訊號才會再買）")
    return {"code": "flat", "label": "空手", "lines": lines}


def _action_winrate(d: pd.DataFrame, sim: dict, held: str) -> dict:
    """高勝率模式（沒有待執行動作時）的續抱／空手說明。"""
    last, pos = d.iloc[-1], sim["position"]
    ok = lambda b: "✓" if b else "✗"
    rsi3 = last["rsi3"]
    if pos:
        left = sg.WIN_MAX_DAYS - pos["days"]
        return {"code": "hold", "label": "續抱", "lines": [
            held + f"，已持有 {pos['days']} 天。",
            f"RSI(3) 目前 {rsi3:.0f}，收盤後高於 {sg.WIN_SELL_ABOVE} → 隔天開盤賣出。",
            f"或再持有 {max(left, 0)} 天（滿 {sg.WIN_MAX_DAYS} 天）就賣出，不論賺賠。",
            "這個模式不設價格停損：歷史測試中加停損反而讓勝率下降。"]}
    ma60 = last[sg.WIN_TREND]
    lines = ["等兩個條件同時成立（今天收盤後判斷，隔天開盤買）：",
             f"{ok(not pd.isna(ma60) and last['close'] > ma60)} 收盤在季線之上（收盤 {_px(last['close'])}，季線 {_px(ma60)}）",
             f"{ok(not pd.isna(rsi3) and rsi3 < sg.WIN_BUY_BELOW)} RSI(3) 低於 {sg.WIN_BUY_BELOW}（目前 {'—' if pd.isna(rsi3) else f'{rsi3:.0f}'}）"]
    if not pd.isna(ma60) and last["close"] <= ma60:
        lines.append("收盤在季線之下：趨勢偏弱，短線再跌也先不接。")
    done = [r for r in sim["rounds"] if r["end_i"] is not None]
    if done:
        r = done[-1]
        day = lambda i: d["date"].iat[i].strftime("%m/%d")
        lines.append(f"上一輪：{day(r['start_i'])} 買進 → {day(r['end_i'])} 賣出（{r['ret']:+.1%}，已扣交易成本）")
    return {"code": "flat", "label": "空手", "lines": lines}


def _rounds_json(rounds: list[dict]) -> list[dict]:
    return [{"s": r["start_i"], "e": r["end_i"], "ret": round(r["ret"], 6), "pnl": round(r["pnl"], 6),
             "mu": r["max_units"], "acts": [{"i": a["i"], "t": a["type"], "px": round(a["px"], 2), "r": a["reason"],
                                             "u": a["units"], "m": a["mode"]} for a in r["actions"]]}
            for r in rounds]


def _signal_list() -> dict:
    groups = ((sg.SIGNALS, "buy", False), (sg.MA_BUY, "buy", True),
              (sg.SELL_SIGNALS, "sell", False), (sg.MA_SELL, "sell", True))
    return {k: {"name": name, "desc": desc, "side": side, "ma": ma}
            for table, side, ma in groups for k, (name, desc) in table.items()}


REGIME_CHAR = {"bull": "b", "bear": "x", "range": "r", "unknown": "u"}


# ---------- 組裝 ----------
def build_stock(daily: pd.DataFrame, overheat: dict | None = None, trading: sg.TradeSettings | None = None,
                market_regime: pd.Series | None = None, code: str = "") -> dict:
    st = trading or sg.TradeSettings(regime_from="self")
    daily = daily.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    # 資料有缺口時分段：每段各自還原拆股、算指標、跑模擬，再接起來
    parts, offset = [], 0
    for seg in ind.split_gaps(daily):
        d_i = ind.add_all(seg, "D")
        sig_i = sg.detect(d_i)
        rg_i = sg.regime(d_i) if market_regime is None else d_i["date"].map(market_regime).fillna("unknown")
        sims_i = {m: sg.simulate(d_i, sig_i, m, rg_i, st, sg.tax_for(code)) for m in sg.MODES}
        parts.append((d_i, sig_i, sg.verdict(d_i, sig_i, **(overheat or {})), rg_i, sims_i, offset))
        offset += len(d_i)
    cat = lambda k: pd.concat([p[k] for p in parts], ignore_index=True)
    d, sig, v, regimes = cat(0), cat(1), cat(2), cat(3)
    daily = d[OHLCV]
    score, sell = sg.buy_score(sig), sg.sell_score(sig)
    sims = {m: sg.merge_sims([(p[4][m], p[5]) for p in parts]) for m in sg.MODES}
    align = sg.ma_alignment(d)
    sim = sims[st.mode]
    regime_now = regimes.iat[-1] if len(regimes) else "unknown"
    mode_now = sg.mode_for(st.mode, regime_now)
    dseries = _series(d)
    dseries["v"] = [int(x) for x in v["level"]]
    dseries["rg"] = "".join(REGIME_CHAR[r] for r in regimes)
    marks = []
    for i in sig.index[sig.any(axis=1)]:
        buys = [k for k in {**sg.SIGNALS, **sg.MA_BUY} if sig.at[i, k]]
        sells = [k for k in {**sg.SELL_SIGNALS, **sg.MA_SELL} if sig.at[i, k]]
        if buys:
            marks.append({"i": int(i), "side": "buy", "s": buys})
        if sells:
            marks.append({"i": int(i), "side": "sell", "s": sells})
    return {
        "D": dseries,
        "W": _series(ind.add_all(ind.resample(daily, "W"), "W")),
        "M": _series(ind.add_all(ind.resample(daily, "M"), "M")),
        "marks": marks,
        "score": int(score.iloc[-1]) if len(score) else 0,
        "sell": int(sell.iloc[-1]) if len(sell) else 0,
        "today": [k for k in sg.ALL_SIGNALS if len(sig) and sig[k].iloc[-5:].any()],
        "verdict": _verdict_now(v, sig),
        "regime": regime_now,
        "modeNow": mode_now,
        "align": align.iat[-1] if len(align) else "unknown",
        "action": _action(d, sig, sim, mode_now, st, align),
        "rounds": _rounds_json(sim["rounds"]),
        "modeStats": {m: sg.round_stats(s["rounds"]) for m, s in sims.items()},
        "hold": float(d["close"].iat[-1] / d["close"].iat[0] - 1) if len(d) > 1 else None,
        "bt": sg.pool_backtests([sg.backtest(p[0], p[1]) for p in parts]),
        "gaps": [int(p[5]) for p in parts[1:]],   # 資料缺口：第幾根 K 棒開始是新的一段
    }


def market_regime_series(prices: pd.DataFrame, code: str) -> pd.Series | None:
    """用某一檔（預設 0050）的年線判斷每天的牛熊，回傳 {日期: regime}；找不到這檔就回傳 None。"""
    segs = [ind.add_all(s, "D") for s in ind.split_gaps(daily_bars_raw(prices, code))]
    if not segs:
        return None
    return pd.concat([pd.Series(sg.regime(d).to_numpy(), index=d["date"]) for d in segs])


def build(out_file: Path = OUT_FILE) -> Path:
    prices = store.load("prices")
    if prices.empty:
        raise SystemExit("沒有資料，先執行 backfill。")
    info = store.load_stock_info()
    watch, basis = pick_watchlist(prices, info, load_watchlist())

    overheat = load_overheat()
    trading = load_trading()
    regime_note = ""
    market = None if trading.regime_from == "self" else market_regime_series(prices, trading.regime_from)
    if trading.regime_from != "self" and market is None:  # 找不到指定的那一檔 → 改用每檔自己的年線，並在畫面說明
        regime_note = f"找不到 {trading.regime_from} 的資料，改用每檔自己的年線判斷牛熊。"
        trading = dataclasses.replace(trading, regime_from="self")
    stocks, backtests = [], []
    for w in watch:
        data = build_stock(daily_bars_raw(prices, w["code"]), overheat, trading, market, w["code"])
        backtests.append(data["bt"])
        stocks.append({**w, **data})

    holds = [s["hold"] for s in stocks if s["hold"] is not None]
    days = sorted(prices["date"].unique())
    payload = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "range": [pd.Timestamp(days[0]).strftime("%Y-%m-%d"), pd.Timestamp(days[-1]).strftime("%Y-%m-%d")],
        "tradingDays": len(days),
        "basis": basis,
        "signals": _signal_list(),
        "alignments": sg.ALIGNMENTS,
        "verdictRule": {"strong": sg.STRONG, "lean": sg.LEAN, "cap": sg.SIGNAL_CAP,
                        "overheatRsi": overheat["overheat_rsi"], "overheatRet5": overheat["overheat_ret5"],
                        "oversoldRsi": overheat["oversold_rsi"]},
        "pooled": sg.pool_backtests(backtests) if backtests else {},
        "modePooled": {m: sg.pool_round_stats([s["modeStats"][m] for s in stocks]) for m in sg.MODES},
        "holdAvg": sum(holds) / len(holds) if holds else None,
        "trading": {"mode": trading.mode, "modes": sg.MODES, "regimeFrom": trading.regime_from,
                    "regimeNote": regime_note,
                    "marketRegime": None if market is None or market.empty else market.iat[-1],
                    "regimes": sg.REGIMES, "maxUnits": trading.max_units, "stop": trading.stop,
                    "maEntry": trading.ma_entry, "maHold": trading.ma_hold, "ma10TurnDays": trading.ma10_turn_days,
                    "entry": sg.ENTRY_RULE, "pullback": sg.PULLBACK_RULE, "winrate": sg.WINRATE_RULE,
                    "fee": trading.fee, "taxStock": sg.TAX_STOCK, "taxEtf": sg.TAX_ETF,
                    "reasons": {k: _reason(k, trading.stop) for k in sg.REASONS}, "actions": sg.ACTIONS},
        "stocks": stocks,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")
    return out_file
