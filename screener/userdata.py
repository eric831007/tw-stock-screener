"""使用者自己的資料：交易紀錄、走勢截圖、自選股代號。

都存在專案的 mydata/ 資料夾（已加進 .gitignore，不會推上 GitHub）：
    mydata/trades.csv                       交易紀錄
    mydata/screenshots/2026-11-05/...png    截圖
    mydata/screenshots/notes.csv            截圖的代號、日期、備註
自選股代號寫回 config.yaml 的 watchlist.extra。
"""
from __future__ import annotations

import datetime as dt
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import signals as sg

ROOT = Path(__file__).resolve().parent.parent
MYDATA = ROOT / "mydata"
CONFIG = ROOT / "config.yaml"
TRADE_COLUMNS = ["date", "code", "side", "price", "shares", "fee", "note"]
# 匯入時接受的欄位名稱（券商匯出的檔案常用中文）
ALIASES = {
    "date": ["date", "日期", "成交日期", "交易日期"],
    "code": ["code", "代號", "股票代號", "證券代號"],
    "side": ["side", "買賣", "買賣別", "交易別"],
    "price": ["price", "價格", "成交價", "成交價格", "成交單價"],
    "shares": ["shares", "股數", "成交股數", "數量"],
    "fee": ["fee", "手續費"],
    "note": ["note", "備註"],
}
BUY_WORDS = {"buy", "b", "買", "買進", "現買", "現股買進"}
SELL_WORDS = {"sell", "s", "賣", "賣出", "現賣", "現股賣出"}
MIN_FEE = 20  # 台股手續費最低 20 元


# ---------- 交易紀錄 ----------
def trades_path() -> Path:
    return MYDATA / "trades.csv"


def write_template(path: Path) -> Path:
    """產生一個空白的交易紀錄範本（含一筆範例），給使用者照著填。"""
    pd.DataFrame([{"日期": "2026-10-14", "代號": "0050", "買賣": "買", "價格": 113.0, "股數": 1000,
                   "手續費": "", "備註": "範例，可刪除"}]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_trades(path: Path) -> pd.DataFrame:
    """讀 CSV 或 Excel，欄位名稱可用中文或英文，回傳標準欄位。"""
    raw = pd.read_excel(path, dtype=str) if path.suffix.lower() in (".xlsx", ".xls") else \
        pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    out = pd.DataFrame()
    for std, names in ALIASES.items():
        col = next((c for c in raw.columns if c.lower() in [n.lower() for n in names]), None)
        out[std] = raw[col] if col else None
    missing = [c for c in ("date", "code", "side", "price", "shares") if out[c].isna().all()]
    if missing:
        raise ValueError(f"找不到欄位：{', '.join(missing)}。需要：日期、代號、買賣、價格、股數")
    out["date"] = pd.to_datetime(out["date"].astype(str).str.replace("/", "-"), errors="coerce")
    out["code"] = out["code"].astype(str).str.strip().str.extract(r"(\d{4,6}[A-Z]?)", expand=False)
    side = out["side"].astype(str).str.strip().str.lower()
    out["side"] = np.where(side.isin(BUY_WORDS), "buy", np.where(side.isin(SELL_WORDS), "sell", None))
    for c in ("price", "shares", "fee"):
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
    out["note"] = out["note"].fillna("").astype(str)
    bad = out[out[["date", "code", "side", "price", "shares"]].isna().any(axis=1)]
    if len(bad):
        raise ValueError(f"第 {', '.join(str(i + 2) for i in bad.index)} 列資料不完整或看不懂（檢查日期、代號、買賣、價格、股數）")
    return out[TRADE_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)


def import_trades(path: Path, replace: bool = False) -> pd.DataFrame:
    """匯入交易紀錄，合併進 mydata/trades.csv（同日、同代號、同買賣、同價、同股數視為重複）。"""
    new = read_trades(path)
    old = load_trades() if (not replace and trades_path().exists()) else pd.DataFrame(columns=TRADE_COLUMNS)
    allt = pd.concat([old, new], ignore_index=True)
    allt = allt.drop_duplicates(subset=["date", "code", "side", "price", "shares"]).sort_values("date", kind="stable")
    save_trades(allt)
    return allt.reset_index(drop=True)


def save_trades(df: pd.DataFrame) -> None:
    MYDATA.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out[TRADE_COLUMNS].to_csv(trades_path(), index=False, encoding="utf-8-sig")


def load_trades() -> pd.DataFrame:
    if not trades_path().exists():
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.read_csv(trades_path(), dtype={"code": str, "note": str}, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["note"] = df["note"].fillna("")
    return df


def costs(side: str, price: float, shares: float, code: str, fee: float | None = None) -> float:
    """這筆交易的手續費（＋賣出時的證交稅），單位元。手續費沒填就照牌價 0.1425%、最低 20 元估算。
    手續費與證交稅都是無條件捨去到元（券商與證交所的算法）。"""
    amount = price * shares
    f = fee if fee is not None and not pd.isna(fee) else max(MIN_FEE, math.floor(amount * sg.FEE))
    return f + (math.floor(amount * sg.tax_for(code)) if side == "sell" else 0)


def positions(trades: pd.DataFrame, last_price: dict[str, float]) -> pd.DataFrame:
    """依交易紀錄算每檔的持股、平均成本（含手續費）、已實現與未實現損益（先進先出）。"""
    rows = []
    for code, g in trades.groupby("code", sort=True):
        lots: list[list[float]] = []      # [剩餘股數, 每股成本]
        realized = 0.0
        for _, t in g.sort_values("date", kind="stable").iterrows():
            c = costs(t.side, t.price, t.shares, code, t.fee)
            if t.side == "buy":
                lots.append([t.shares, (t.price * t.shares + c) / t.shares])
            else:
                left, proceeds = t.shares, t.price * t.shares - c
                cost_basis = 0.0
                while left > 0 and lots:
                    take = min(left, lots[0][0])
                    cost_basis += take * lots[0][1]
                    lots[0][0] -= take
                    left -= take
                    if lots[0][0] <= 0:
                        lots.pop(0)
                realized += proceeds - cost_basis
        held = sum(l[0] for l in lots)
        avg = sum(l[0] * l[1] for l in lots) / held if held else float("nan")
        px = last_price.get(code, float("nan"))
        unreal = (px * held - costs("sell", px, held, code) - avg * held) if held and not np.isnan(px) else 0.0
        rows.append({"code": code, "shares": held, "avg_cost": avg, "last": px, "realized": realized,
                     "unrealized": unreal, "unrealized_pct": unreal / (avg * held) if held else float("nan")})
    return pd.DataFrame(rows, columns=["code", "shares", "avg_cost", "last", "realized", "unrealized", "unrealized_pct"])


def compare_with_rule(trades: pd.DataFrame, rule_actions: dict[str, list[tuple[pd.Timestamp, str]]],
                      window: int = 3) -> pd.Series:
    """每筆交易跟規則比：前後 window 個日曆天內，規則有沒有同方向的成交。回傳每筆的說明文字。"""
    out = []
    for _, t in trades.iterrows():
        acts = rule_actions.get(t.code, [])
        want = "buy" if t.side == "buy" else "sell"
        near = [d for d, a in acts if a == want and abs((d - t.date).days) <= window]
        other = [d for d, a in acts if a != want and abs((d - t.date).days) <= window]
        if near:
            out.append(f"符合規則（規則 {near[0]:%m/%d} 也{'買' if want == 'buy' else '賣'}）")
        elif other:
            out.append(f"跟規則相反（規則 {other[0]:%m/%d} 是{'賣' if want == 'buy' else '買'}）")
        else:
            out.append("規則當時沒有訊號")
    return pd.Series(out, index=trades.index, dtype=object)


# ---------- 截圖 ----------
def notes_path() -> Path:
    return MYDATA / "screenshots" / "notes.csv"


def add_screenshot(src: Path, code: str, day: dt.date, note: str = "") -> Path:
    """把截圖複製進 mydata/screenshots/日期/，並記下代號與備註。"""
    folder = MYDATA / "screenshots" / day.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%H%M%S")
    dst = folder / f"{code or 'market'}_{stamp}{src.suffix.lower()}"
    i = 1
    while dst.exists():
        dst = folder / f"{code or 'market'}_{stamp}_{i}{src.suffix.lower()}"
        i += 1
    shutil.copy2(src, dst)
    df = list_screenshots()
    df = pd.concat([df, pd.DataFrame([{"date": day.isoformat(), "code": code, "file": str(dst.relative_to(MYDATA)),
                                       "note": note}])], ignore_index=True)
    df.to_csv(notes_path(), index=False, encoding="utf-8-sig")
    return dst


def list_screenshots(code: str | None = None) -> pd.DataFrame:
    if not notes_path().exists():
        return pd.DataFrame(columns=["date", "code", "file", "note"])
    df = pd.read_csv(notes_path(), dtype=str, encoding="utf-8-sig").fillna("")
    return df[df["code"] == code] if code else df


# ---------- 自選股 ----------
def extra_codes() -> list[str]:
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    return [str(c) for c in (cfg.get("watchlist", {}) or {}).get("extra", []) or []]


def set_extra_codes(codes: list[str]) -> None:
    """改寫 config.yaml 的 watchlist.extra 那一行，保留其他內容與註解。"""
    text = CONFIG.read_text(encoding="utf-8")
    clean = []
    for c in codes:
        c = str(c).strip()
        if re.fullmatch(r"\d{4,6}[A-Z]?", c) and c not in clean:
            clean.append(c)
    new_line = "  extra: [" + ", ".join(f'"{c}"' for c in clean) + "]"
    pattern = re.compile(r"^  extra:\s*\[[^\]]*\](?P<comment>[ \t]*#.*)?$", re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError("config.yaml 找不到 watchlist.extra 這一行")
    comment = (m.group("comment") or "").strip()
    text = text[:m.start()] + (f"{new_line.ljust(30)}{comment}" if comment else new_line) + text[m.end():]
    CONFIG.write_text(text, encoding="utf-8")
