"""每支股票各自找「勝率最高」的高勝率規則參數，並用沒看過的期間驗證。

    python -m screener optimize                       # 預設：2025-01-01 之前挑規則，之後驗證
    python -m screener optimize --split 2026-01-01    # 改用別的切分日

做法（避免「挑到運氣好的參數」）：
  1. 挑選期（split 之前）：每支股票試 GRID 裡的每一組參數，只留下
     交易至少 MIN_TRADES 筆、扣成本後平均報酬 > 0 的，從中選勝率最高的（同分看平均報酬）。
  2. 驗證期（split 之後）：用選出的參數實際跑一次，跟「共用規則」（預設參數）在同一支股票上比較。
  3. 判定「適用」：驗證期至少 MIN_TEST_TRADES 筆、勝率不低於共用規則、平均報酬 > 0。
     其他標「不適用」或「樣本不足」，這些股票繼續用共用規則。

結果寫到 reports/per_stock_rules.csv（給程式讀）與 reports/per_stock_rules.md（給人看）。
資料有缺口（相鄰兩個交易日差超過 GAP_DAYS 天）時會切成好幾段分開算，避免指標跨過缺口。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import dashboard
from . import indicators as ind
from . import signals as sg
from . import store

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
GAP_DAYS = 15
MIN_TRADES = 8          # 挑選期至少幾筆交易
MIN_TEST_TRADES = 3     # 驗證期至少幾筆才判斷適不適用
GRID = {
    "win_trend": ["none", "ma20", "ma60", "ma120"],
    "win_osc": ["rsi2", "rsi3"],
    "win_buy_below": [10, 15, 20, 25, 30],
    "win_sell_above": [60, 70, 80],
    "win_max_days": [10, 20],
}


def segments(daily: pd.DataFrame, gap_days: int = GAP_DAYS, min_len: int = 30) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """依資料缺口切段，每段各自還原拆股、計算指標與訊號，回傳 [(日線, 訊號)]。"""
    out = []
    for seg in ind.split_gaps(daily, gap_days):
        if len(seg) >= min_len:
            d = ind.add_all(seg)
            out.append((d, sg.detect(d)))
    return out


def trades(segs: list[tuple[pd.DataFrame, pd.DataFrame]], st: sg.TradeSettings,
           tax: float) -> list[tuple[pd.Timestamp, float]]:
    """所有已完成的交易：(買進日, 扣成本後報酬)。"""
    out = []
    for d, sig in segs:
        sim = sg.simulate(d, sig, "winrate", settings=st, tax=tax)
        out += [(d["date"].iat[r["start_i"]], r["ret"]) for r in sim["rounds"] if r["end_i"] is not None]
    return out


def summarize(rets: list[float]) -> dict:
    if not rets:
        return {"n": 0, "win": None, "avg": None, "worst": None}
    a = np.array(rets)
    return {"n": int(len(a)), "win": float((a > 0).mean()), "avg": float(a.mean()), "worst": float(a.min())}


def _split(tr: list[tuple[pd.Timestamp, float]], split: pd.Timestamp) -> tuple[list[float], list[float]]:
    return [r for d, r in tr if d < split], [r for d, r in tr if d >= split]


def candidates(base: sg.TradeSettings = sg.TradeSettings()) -> list[sg.TradeSettings]:
    keys = list(GRID)
    return [dataclasses.replace(base, **dict(zip(keys, combo))) for combo in itertools.product(*GRID.values())]


def optimize_stock(segs: list[tuple[pd.DataFrame, pd.DataFrame]], code: str, split: pd.Timestamp,
                   grid: list[sg.TradeSettings] | None = None) -> dict:
    """一支股票：挑選期找最佳參數，驗證期跟共用規則比較。"""
    tax = sg.tax_for(code)
    common = sg.TradeSettings()
    best, best_key = None, None
    for st in grid or candidates():
        train, _ = _split(trades(segs, st, tax), split)
        s = summarize(train)
        if s["n"] < MIN_TRADES or s["avg"] <= 0:
            continue
        key = (s["win"], s["avg"])
        if best_key is None or key > best_key:
            best, best_key = st, key
    c_train, c_test = (summarize(x) for x in _split(trades(segs, common, tax), split))
    out = {"common_train": c_train, "common_test": c_test, "best": best}
    if best is None:
        out.update(train=None, test=None, verdict="找不到")
        return out
    b_train, b_test = (summarize(x) for x in _split(trades(segs, best, tax), split))
    out.update(train=b_train, test=b_test)
    if b_test["n"] < MIN_TEST_TRADES:
        out["verdict"] = "樣本不足"
    elif b_test["avg"] > 0 and (c_test["win"] is None or b_test["win"] >= c_test["win"]):
        out["verdict"] = "適用"
    else:
        out["verdict"] = "不適用"
    return out


def _pct(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.1%}" if signed else f"{v:.0%}"


def run(split: dt.date = dt.date(2025, 1, 1), out_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    prices = store.load("prices")
    if prices.empty:
        raise SystemExit("沒有資料，先執行 backfill。")
    watch, _ = dashboard.pick_watchlist(prices, store.load_stock_info(), dashboard.load_watchlist())
    split_ts = pd.Timestamp(split)
    grid = candidates()
    rows = []
    for w in watch:
        segs = segments(dashboard.daily_bars_raw(prices, w["code"]))
        r = optimize_stock(segs, w["code"], split_ts, grid)
        b = r["best"]
        row = {"code": w["code"], "name": w["name"], "verdict": r["verdict"]}
        for k in GRID:
            row[k] = getattr(b, k) if b else None
        for tag, s in (("train", r["train"]), ("test", r["test"]),
                       ("common_train", r["common_train"]), ("common_test", r["common_test"])):
            for f in ("n", "win", "avg", "worst"):
                row[f"{tag}_{f}"] = None if s is None else s[f]
        row["rule"] = sg.winrate_rule(b) if b else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "per_stock_rules.csv", out_dir / "per_stock_rules.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(_markdown(df, prices, split), encoding="utf-8")
    return csv_path, md_path


def _markdown(df: pd.DataFrame, prices: pd.DataFrame, split: dt.date) -> str:
    days = sorted(prices["date"].unique())
    first, last = pd.Timestamp(days[0]).date(), pd.Timestamp(days[-1]).date()
    ok = df[df["verdict"] == "適用"]

    def pooled(prefix: str, rows: pd.DataFrame) -> str:
        n = rows[f"{prefix}_n"].fillna(0)
        tot = int(n.sum())
        if not tot:
            return "—"
        win = (rows[f"{prefix}_win"].fillna(0) * n).sum() / tot
        avg = (rows[f"{prefix}_avg"].fillna(0) * n).sum() / tot
        return f"{tot} 筆，勝率 {win:.0%}，平均 {avg:+.2%}"

    lines = [
        "# 每支股票的最高勝率規則", "",
        f"產生時間：{dt.datetime.now():%Y-%m-%d %H:%M}　資料期間：{first} ～ {last}（{len(days)} 個交易日）",
        f"挑選期：{first} ～ {split - dt.timedelta(days=1)}　驗證期：{split} ～ {last}", "",
        "> 每支股票只用挑選期找參數，驗證期是「沒看過的資料」。驗證期勝率不低於共用規則、而且平均賺錢，才標「適用」。",
        "> 其他股票繼續用共用規則。資料補齊後重跑 `python -m screener optimize` 再檢查一次。", "",
        "## 整體比較（驗證期）", "",
        "| | 全部股票 | 只看「適用」的股票 |", "| --- | --- | --- |",
        f"| 各自最佳規則 | {pooled('test', df)} | {pooled('test', ok)} |",
        f"| 共用規則 | {pooled('common_test', df)} | {pooled('common_test', ok)} |", "",
        f"適用 {len(ok)} 檔、不適用 {(df['verdict'] == '不適用').sum()} 檔、"
        f"樣本不足 {(df['verdict'] == '樣本不足').sum()} 檔、找不到 {(df['verdict'] == '找不到').sum()} 檔。", "",
        "## 各股票", "",
        "| 代號 | 名稱 | 判定 | 專屬規則 | 挑選期 筆數／勝率／平均 | 驗證期 筆數／勝率／平均／最差 | 共用規則驗證期 筆數／勝率／平均 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = {"適用": 0, "不適用": 1, "樣本不足": 2, "找不到": 3}
    for _, r in df.sort_values(["verdict", "code"], key=lambda s: s.map(order) if s.name == "verdict" else s).iterrows():
        tr = f"{r.train_n:.0f}／{_pct(r.train_win)}／{_pct(r.train_avg, True)}" if pd.notna(r.train_n) else "—"
        te = (f"{r.test_n:.0f}／{_pct(r.test_win)}／{_pct(r.test_avg, True)}／{_pct(r.test_worst, True)}"
              if pd.notna(r.test_n) else "—")
        ct = f"{r.common_test_n:.0f}／{_pct(r.common_test_win)}／{_pct(r.common_test_avg, True)}"
        lines.append(f"| {r.code} | {r['name']} | {r.verdict} | {r.rule or '—'} | {tr} | {te} | {ct} |")
    lines += ["", "共用規則：" + sg.WINRATE_RULE + "。所有報酬已扣手續費與證交稅。", ""]
    return "\n".join(lines)
