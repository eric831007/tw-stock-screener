"""桌面程式的資料與分析（不含任何視窗程式碼，方便測試）。

- 讀取：專案 data/ ＋ 歷史資料夾（自動找 ../tw-stock-history/data，或環境變數 SCREENER_EXTRA_DATA_DIRS）
- 名單：config.yaml 的自選股＋權值股，再加上使用者自己加入的代號
- 分析：共用的高勝率規則（季線之上 RSI(3) < 25 買；RSI(3) > 70 或滿 20 天賣）
- 盤中：證交所即時報價，試算「如果收在現價」會不會出現買賣訊號
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import dashboard
from . import indicators as ind
from . import realtime
from . import signals as sg
from . import store
from . import userdata

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = ROOT.parent / "tw-stock-history" / "data"


def use_default_history() -> list[Path]:
    """沒有用環境變數指定歷史資料夾時，自動加入 ../tw-stock-history/data（存在的話）。"""
    if not store.EXTRA_DIRS and DEFAULT_HISTORY.exists():
        store.EXTRA_DIRS.append(DEFAULT_HISTORY)
    return list(store.EXTRA_DIRS)


@dataclass
class StockState:
    code: str
    name: str
    market: str
    group: str
    d: pd.DataFrame                     # 日 K ＋ 指標（各段分開計算後接起來）
    sim: dict                           # 高勝率規則的模擬結果（各段接起來）
    action: dict                        # 最新一天的操作建議 {code, label, lines}
    gaps: list[int] = field(default_factory=list)
    last_segment: pd.DataFrame | None = None   # 最後一段的原始日 K（盤中試算用）
    chips: dict = field(default_factory=dict)  # 三大法人籌碼摘要（見 chip_summary）

    @property
    def last_close(self) -> float:
        return float(self.d["close"].iat[-1]) if len(self.d) else float("nan")

    def rule_actions(self) -> list[tuple[pd.Timestamp, str]]:
        """規則的每一次買／賣（成交日），給交易紀錄比對用。"""
        out = []
        for r in self.sim["rounds"]:
            for a in r["actions"]:
                out.append((self.d["date"].iat[a["i"]], "buy" if a["type"] in ("buy", "add") else "sell"))
        return out


INST_COLS = {"foreign_net": "外資", "trust_net": "投信", "dealer_net": "自營商", "total_net": "三大法人合計"}


def _streak(s: pd.Series) -> int:
    """最後連續幾天同方向：正數＝連續買超天數，負數＝連續賣超天數，0＝最後一天沒買賣。"""
    v = s.dropna().to_numpy()
    if not len(v) or v[-1] == 0:
        return 0
    sign, n = (1 if v[-1] > 0 else -1), 0
    for x in v[::-1]:
        if (1 if x > 0 else -1 if x < 0 else 0) != sign:
            break
        n += 1
    return sign * n


MARGIN_COLS = ["margin_bal", "short_bal", "margin_buy", "margin_sell"]
HOLDER_COLS = ["pct_1000", "pct_400", "pct_le50", "holders", "holders_1000"]


def _change(s: pd.Series, n: int) -> float:
    v = s.dropna()
    return float(v.iat[-1] - v.iat[-1 - n]) if len(v) > n else float("nan")


def chip_summary(d: pd.DataFrame) -> dict:
    """籌碼摘要：
    三大法人：連續買賣超天數、近 5／20 日累計（張）、近 20 日買超佔成交量比例；
    融資融券：最新餘額、近 5／20 日增減、券資比；集保：千張大戶比例與週變化。"""
    out = {}
    if "margin_bal" in d and d["margin_bal"].notna().any():
        mb, sb = d["margin_bal"].dropna(), d["short_bal"].dropna()
        out["margin"] = {"bal": float(mb.iat[-1]), "chg5": _change(d["margin_bal"], 5),
                         "chg20": _change(d["margin_bal"], 20), "short": float(sb.iat[-1]) if len(sb) else float("nan"),
                         "short_chg5": _change(d["short_bal"], 5),
                         "ratio": float(sb.iat[-1] / mb.iat[-1]) if len(sb) and mb.iat[-1] else float("nan"),
                         "date": d.loc[mb.index[-1], "date"]}
    if "pct_1000" in d and d["pct_1000"].notna().any():
        w = d.dropna(subset=["pct_1000"]).drop_duplicates("holders_date")
        out["holders"] = {"pct_1000": float(w["pct_1000"].iat[-1]), "chg1": _change(w["pct_1000"], 1),
                          "chg4": _change(w["pct_1000"], 4), "pct_le50": float(w["pct_le50"].iat[-1]),
                          "n_1000": float(w["holders_1000"].iat[-1]), "weeks": len(w),
                          "date": w["holders_date"].iat[-1]}
    if "foreign_net" not in d or d["foreign_net"].dropna().empty:
        return out
    vol20 = d["volume"].iloc[-20:].sum()
    for col, name in INST_COLS.items():
        s = d[col].fillna(0) if col in d else pd.Series(dtype=float)
        out[col] = {"name": name, "streak": _streak(s), "sum5": s.iloc[-5:].sum() / 1000,
                    "sum20": s.iloc[-20:].sum() / 1000,
                    "ratio20": s.iloc[-20:].sum() / vol20 if vol20 else float("nan")}
    return out


def chip_lines(chips: dict) -> list[str]:
    lines = []
    m = chips.get("margin")
    if m:
        def sgn(v):
            return "—" if pd.isna(v) else f"{v:+,.0f}"
        lines.append(f"融資餘額 {m['bal']:,.0f} 張（近 5 日 {sgn(m['chg5'])}、近 20 日 {sgn(m['chg20'])}）；"
                     f"融券餘額 {m['short']:,.0f} 張（近 5 日 {sgn(m['short_chg5'])}）；券資比 {m['ratio']:.1%}"
                     f"（{m['date']:%m/%d}）")
    h = chips.get("holders")
    if h:
        def pp(v):
            return "—" if pd.isna(v) else f"{v:+.2f}"
        lines.append(f"千張大戶持股 {h['pct_1000']:.2f}%（較上週 {pp(h['chg1'])}、較 4 週前 {pp(h['chg4'])} 個百分點），"
                     f"{h['n_1000']:,.0f} 人；50 張以下散戶 {h['pct_le50']:.2f}%（集保 {h['date']:%m/%d}，已累積 {h['weeks']} 週）")
    for col in ("foreign_net", "trust_net", "dealer_net"):
        c = chips.get(col)
        if not c:
            continue
        st = c["streak"]
        streak = f"連續買超 {st} 天" if st > 0 else f"連續賣超 {-st} 天" if st < 0 else "最近一天沒有買賣"
        lines.append(f"{c['name']}：{streak}；近 5 日 {c['sum5']:+,.0f} 張、近 20 日 {c['sum20']:+,.0f} 張"
                     f"（佔成交量 {c['ratio20']:+.1%}）")
    return lines


class Model:
    def __init__(self, settings: sg.TradeSettings | None = None):
        use_default_history()
        self.settings = settings or sg.TradeSettings(mode="winrate")
        self.cache: dict[str, StockState] = {}
        self.quotes = pd.DataFrame()
        self.reload()

    # ---------- 資料 ----------
    def reload(self) -> None:
        self.prices = store.load("prices")
        self.inst = store.load("institutional")
        self.margin = store.load("margin")
        self.holders = store.load(store.HOLDERS)
        self.cache.clear()
        if self.prices.empty:
            self.latest = pd.DataFrame()
            self.watch = []
            return
        self.latest = self.prices[self.prices["date"] == self.prices["date"].max()].set_index("code")
        cfg = dashboard.load_watchlist()
        self.watch, self.basis = dashboard.pick_watchlist(self.prices, store.load_stock_info(), cfg)
        # pick_watchlist 已經包含 extra；這裡確保 extra 裡最新一天沒有成交的代號也列出來
        listed = {w["code"] for w in self.watch}
        names = self.prices.drop_duplicates("code", keep="last").set_index("code")["name"]
        for c in userdata.extra_codes():
            if c not in listed and c in names.index:
                self.watch.insert(0, {"code": c, "name": names[c], "group": "自選"})

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self.prices.empty:
            return None
        return self.prices["date"].min(), self.prices["date"].max()

    def market_of(self, code: str) -> str:
        if code in self.latest.index:
            return str(self.latest.at[code, "market"])
        rows = self.prices[self.prices["code"] == code]
        return str(rows["market"].iat[-1]) if len(rows) else "TWSE"

    def known_code(self, code: str) -> bool:
        return bool((self.prices["code"] == code).any())

    # ---------- 分析 ----------
    def analyze(self, code: str) -> StockState:
        if code in self.cache:
            return self.cache[code]
        w = next((x for x in self.watch if x["code"] == code), {"code": code, "name": code, "group": ""})
        raw = dashboard.daily_bars_raw(self.prices, code)
        segs = ind.split_gaps(raw)
        tax = sg.tax_for(code)
        parts, ds, offset = [], [], 0
        for seg in segs:
            d_i = ind.add_all(seg)
            sig_i = sg.detect(d_i)
            parts.append((sg.simulate(d_i, sig_i, "winrate", settings=self.settings, tax=tax), offset))
            ds.append((d_i, sig_i))
            offset += len(d_i)
        if ds:
            d = pd.concat([x[0] for x in ds], ignore_index=True)
            sig = pd.concat([x[1] for x in ds], ignore_index=True)
        else:
            d, sig = ind.add_all(raw), sg.detect(ind.add_all(raw))
        sim = sg.merge_sims(parts) if parts else {"rounds": [], "pending": None, "position": None}
        action = dashboard._action(d, sig, sim, "winrate", self.settings, sg.ma_alignment(d)) if len(d) else \
            {"code": "flat", "label": "無資料", "lines": ["沒有這檔的日資料"]}
        gaps = [p[1] for p in parts[1:]]
        d = self._with_chips(self._with_inst(d, code), code)
        st = StockState(code, w["name"], self.market_of(code), w.get("group", ""), d, sim, action, gaps,
                        segs[-1] if segs else raw, chip_summary(d))
        self.cache[code] = st
        return st

    def _with_inst(self, d: pd.DataFrame, code: str) -> pd.DataFrame:
        """把三大法人買賣超（股）併進日資料；沒有資料的日子是 NaN。"""
        if self.inst.empty or d.empty:
            return d.assign(**{c: float("nan") for c in INST_COLS})
        inst = self.inst[self.inst["code"] == code][["date", *INST_COLS]].drop_duplicates("date", keep="last")
        return d.merge(inst, on="date", how="left")

    def _with_chips(self, d: pd.DataFrame, code: str) -> pd.DataFrame:
        """併入融資融券（每天）與集保（每週，往後沿用到下一週資料出現）。"""
        if d.empty:
            return d
        if not self.margin.empty:
            m = self.margin[self.margin["code"] == code][["date", *MARGIN_COLS]].drop_duplicates("date", keep="last")
            d = d.merge(m, on="date", how="left")
        else:
            d = d.assign(**{c: float("nan") for c in MARGIN_COLS})
        if not self.holders.empty:
            h = self.holders[self.holders["code"] == code][["date", *HOLDER_COLS]].sort_values("date")
            h = h.rename(columns={"date": "holders_date"})
            d = pd.merge_asof(d.sort_values("date"), h, left_on="date", right_on="holders_date", direction="backward")
        else:
            d = d.assign(holders_date=pd.NaT, **{c: float("nan") for c in HOLDER_COLS})
        return d.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """名單上每一檔的最新收盤與操作建議（第一次會花幾秒計算）。"""
        rows = []
        for w in self.watch:
            s = self.analyze(w["code"])
            prev = s.d["close"].iat[-2] if len(s.d) > 1 else float("nan")
            rows.append({"code": s.code, "name": s.name, "group": s.group, "close": s.last_close,
                         "change_pct": s.last_close / prev - 1 if prev else float("nan"),
                         "action": s.action["label"], "action_code": s.action["code"]})
        return pd.DataFrame(rows)

    # ---------- 盤中 ----------
    def fetch_quotes(self) -> pd.DataFrame:
        codes = {w["code"]: self.market_of(w["code"]) for w in self.watch}
        self.quotes = realtime.fetch_quotes(codes)
        return self.quotes

    def intraday(self, code: str) -> dict | None:
        if self.quotes.empty or code not in self.quotes.index:
            return None
        s = self.analyze(code)
        return realtime.intraday_check(s.last_segment, self.quotes.loc[code], self.settings, sg.tax_for(code))

    # ---------- 使用者資料 ----------
    def add_codes(self, codes: list[str]) -> tuple[list[str], list[str]]:
        """加入自選股，回傳（加入的, 找不到資料的）。"""
        ok = [c for c in codes if self.known_code(c)]
        bad = [c for c in codes if c not in ok]
        if ok:
            userdata.set_extra_codes(userdata.extra_codes() + ok)
            self.reload()
        return ok, bad

    def remove_code(self, code: str) -> None:
        userdata.set_extra_codes([c for c in userdata.extra_codes() if c != code])
        self.reload()

    def my_trades(self) -> pd.DataFrame:
        t = userdata.load_trades()
        if t.empty:
            return t.assign(compare=pd.Series(dtype=object))
        acts = {c: self.analyze(c).rule_actions() for c in t["code"].unique() if self.known_code(c)}
        return t.assign(compare=userdata.compare_with_rule(t, acts))

    def my_positions(self) -> pd.DataFrame:
        t = userdata.load_trades()
        if t.empty:
            return userdata.positions(t, {})
        last = {}
        for c in t["code"].unique():
            if not self.quotes.empty and c in self.quotes.index and pd.notna(self.quotes.at[c, "price"]):
                last[c] = float(self.quotes.at[c, "price"])
            elif self.known_code(c):
                last[c] = self.analyze(c).last_close
        return userdata.positions(t, last)


def git_pull() -> str:
    """從 GitHub 拉最新資料（只快轉，不會覆蓋本機修改）。回傳 git 的輸出。"""
    r = subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only", "origin", "main"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).strip() or ("完成" if r.returncode == 0 else f"失敗（代碼 {r.returncode}）")
