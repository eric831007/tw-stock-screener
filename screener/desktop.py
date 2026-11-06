"""台股分析工具（桌面視窗程式）。

    .venv\\python.exe -m screener app      或直接點兩下專案裡的「啟動分析工具.bat」

左邊是股票名單與操作建議；右邊是走勢圖（K 線、季線、買賣標記、RSI(3)）、操作建議與盤中試算、
自己的交易紀錄、截圖。盤中報價每分鐘自動更新（9:00–13:30），也可以按「立即更新報價」。
截圖可以用 Win+Shift+S 截完後在視窗裡按 Ctrl+V 貼上，或把圖片檔拖進視窗。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import app_model, realtime, userdata
from . import indicators as ind

UP, DOWN, FLAT = "#d93b3b", "#1f9d55", "#888888"   # 台股：紅漲綠跌
ACTION_COLOR = {"buy": UP, "add": UP, "sell": DOWN, "half": DOWN, "hold": "#2f6fdb", "flat": FLAT}
TZ = dt.timezone(dt.timedelta(hours=8))
VIEW_BARS = 250
COLS = ["代號", "名稱", "價格", "漲跌", "建議", "盤中試算"]


def fmt(v, d=2) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"


def pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2%}"


# ---------- 背景工作（避免查報價、計算時畫面卡住） ----------
class _Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class Job(QtCore.QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args, self.signals = fn, args, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(*self.args))
        except Exception:
            self.signals.failed.emit(traceback.format_exc(limit=3))


# ---------- 圖表元件 ----------
class CandleItem(pg.GraphicsObject):
    """K 棒：紅漲綠跌；provisional=True 時畫成空心（盤中尚未收盤）。"""

    def __init__(self, x, o, h, l, c, provisional=False):
        super().__init__()
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        w = 0.35
        for xi, oi, hi, li, ci in zip(x, o, h, l, c):
            if np.isnan(ci):
                continue
            col = QtGui.QColor(UP if ci >= oi else DOWN)
            pen = pg.mkPen(col, width=1, style=QtCore.Qt.DashLine if provisional else QtCore.Qt.SolidLine)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(xi, li), QtCore.QPointF(xi, hi))
            p.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush) if provisional else pg.mkBrush(col))
            p.drawRect(QtCore.QRectF(xi - w, min(oi, ci), 2 * w, max(abs(ci - oi), 1e-6)))
        p.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class DateAxis(pg.AxisItem):
    def __init__(self, dates, **kw):
        super().__init__(orientation="bottom", **kw)
        self.dates = dates

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            i = int(round(v))
            out.append(self.dates[i].strftime("%y-%m-%d") if 0 <= i < len(self.dates) else "")
        return out


MA_COLORS = ["#e0a100", "#9b51e0", "#2f6fdb", "#0f9d8a", "#c2410c", "#6b7280"]
MA_NAMES = {"D": {5: "5日", 10: "10日", 20: "月線", 60: "季線", 120: "半年線", 240: "年線"},
            "W": {5: "5週", 10: "10週", 20: "20週", 52: "52週"},
            "M": {3: "3月", 6: "6月", 12: "12月", 24: "24月"}}
MA_DEFAULT = {"D": {5, 20, 60}, "W": {5, 20, 52}, "M": {6, 12, 24}}   # 日 K 預設：5日、月線、季線
PANES = {"rule": "RSI(3)（買賣規則）", "vol": "成交量", "inst": "三大法人", "margin": "融資融券",
         "holders": "千張大戶", "macd": "MACD", "kd": "KD", "rsi": "RSI(14)"}
PANES_DEFAULT = {"rule", "vol"}          # 預設只開 RSI(3) 與成交量，其他自己勾
INST = [("foreign_net", "外資", "#2f6fdb"), ("trust_net", "投信", "#c2410c"), ("dealer_net", "自營商", "#9b51e0")]


class Chart(QtWidgets.QWidget):
    """走勢圖：日／週／月 K、均線（可開關）、布林通道、買賣標籤，下方可選 RSI(3)、成交量、三大法人、MACD、KD、RSI(14)。
    滑鼠移到圖上會在上方顯示那一天的開高低收、均線與指標數值。"""

    def __init__(self):
        super().__init__()
        self.tf = "D"
        self.ma_on = {k: set(v) for k, v in MA_DEFAULT.items()}
        self.panes_on = set(PANES_DEFAULT)
        self.bb_on = False
        self.buys_on = True        # 買進標籤：預設顯示
        self.sells_on = False      # 賣出標籤：預設不顯示
        self._load_settings()
        self.state = None
        self.bar = None
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        bar1 = QtWidgets.QHBoxLayout()
        self.tf_group = QtWidgets.QButtonGroup(self)
        for key, label in (("D", "日K"), ("W", "週K"), ("M", "月K")):
            b = QtWidgets.QPushButton(label, checkable=True, checked=key == "D")
            b.setFixedWidth(52)
            b.clicked.connect(lambda _=False, k=key: self._set_tf(k))
            self.tf_group.addButton(b)
            bar1.addWidget(b)
        bar1.addSpacing(10)
        self.ma_box = QtWidgets.QHBoxLayout()
        bar1.addLayout(self.ma_box)
        self.bb_check = QtWidgets.QCheckBox("布林通道", checked=self.bb_on)
        self.bb_check.toggled.connect(self._toggle_bb)
        bar1.addWidget(self.bb_check)
        self.buy_check = QtWidgets.QCheckBox("買進標籤", checked=self.buys_on)
        self.buy_check.setStyleSheet(f"QCheckBox {{ color: {UP}; font-weight: 600; }}")
        self.buy_check.toggled.connect(lambda on: self._toggle_marks("buy", on))
        bar1.addWidget(self.buy_check)
        self.sell_check = QtWidgets.QCheckBox("賣出標籤", checked=self.sells_on)
        self.sell_check.setStyleSheet(f"QCheckBox {{ color: {DOWN}; font-weight: 600; }}")
        self.sell_check.toggled.connect(lambda on: self._toggle_marks("sell", on))
        bar1.addWidget(self.sell_check)
        bar1.addStretch(1)
        v.addLayout(bar1)
        bar2 = QtWidgets.QHBoxLayout()
        bar2.addWidget(QtWidgets.QLabel("下方指標："))
        for key, label in PANES.items():
            c = QtWidgets.QCheckBox(label, checked=key in self.panes_on)
            c.toggled.connect(lambda on, k=key: self._toggle_pane(k, on))
            bar2.addWidget(c)
        bar2.addStretch(1)
        v.addLayout(bar2)
        self.info = QtWidgets.QLabel(" ")
        self.info.setTextFormat(QtCore.Qt.RichText)
        self.info.setStyleSheet("padding: 2px 4px; font-size: 12px;")
        v.addWidget(self.info)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        v.addWidget(self.glw, 1)
        self._build_ma_checks()

    # ----- 記住勾選（mydata/chart_settings.json；刪掉這個檔就恢復預設） -----
    @staticmethod
    def settings_path() -> Path:
        return userdata.MYDATA / "chart_settings.json"

    def _load_settings(self):
        try:
            cfg = json.loads(self.settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for tf, ps in (cfg.get("ma_on") or {}).items():
            if tf in self.ma_on:
                self.ma_on[tf] = {int(p) for p in ps if int(p) in MA_NAMES[tf]}
        if isinstance(cfg.get("panes_on"), list):
            self.panes_on = {k for k in cfg["panes_on"] if k in PANES}
        self.bb_on = bool(cfg.get("bb_on", self.bb_on))
        self.buys_on = bool(cfg.get("buys_on", self.buys_on))
        self.sells_on = bool(cfg.get("sells_on", self.sells_on))

    def _save_settings(self):
        cfg = {"ma_on": {tf: sorted(ps) for tf, ps in self.ma_on.items()}, "panes_on": sorted(self.panes_on),
               "bb_on": self.bb_on, "buys_on": self.buys_on, "sells_on": self.sells_on}
        try:
            userdata.MYDATA.mkdir(parents=True, exist_ok=True)
            self.settings_path().write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass   # 存不了就算了，不影響使用

    # ----- 控制項 -----
    def _build_ma_checks(self):
        while self.ma_box.count():
            w = self.ma_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        for j, (p, name) in enumerate(MA_NAMES[self.tf].items()):
            c = QtWidgets.QCheckBox(name, checked=p in self.ma_on[self.tf])
            c.setStyleSheet(f"QCheckBox {{ color: {MA_COLORS[j]}; font-weight: 600; }}")
            c.toggled.connect(lambda on, p=p: self._toggle_ma(p, on))
            self.ma_box.addWidget(c)

    def _set_tf(self, tf):
        self.tf = tf
        self._build_ma_checks()
        self.redraw()

    def _toggle_ma(self, p, on):
        (self.ma_on[self.tf].add if on else self.ma_on[self.tf].discard)(p)
        self._save_settings()
        self.redraw()

    def _toggle_bb(self, on):
        self.bb_on = on
        self._save_settings()
        self.redraw()

    def _toggle_marks(self, side, on):
        if side == "buy":
            self.buys_on = on
        else:
            self.sells_on = on
        self._save_settings()
        self.redraw()

    def _toggle_pane(self, key, on):
        (self.panes_on.add if on else self.panes_on.discard)(key)
        self._save_settings()
        self.redraw()

    # ----- 資料 -----
    def show_stock(self, s, intraday_bar: dict | None = None):
        self.state, self.bar = s, intraday_bar
        self.redraw()

    def _frame(self) -> pd.DataFrame:
        d = self.state.d
        if self.tf == "D":
            return d.reset_index(drop=True)
        base = d[["date", "open", "high", "low", "close", "volume"]]
        return ind.add_all(ind.resample(base, self.tf), self.tf)

    def _trade_marks(self, f: pd.DataFrame):
        """規則的買賣，對應到目前 K 棒（週／月 K 對到那一週／月）。回傳 [(x, 'buy'/'sell', 文字)]。"""
        s, out = self.state, []
        bar_dates = f["date"].to_numpy()
        for r in s.sim["rounds"]:
            for a in r["actions"]:
                day = s.d["date"].iat[a["i"]]
                x = int(np.searchsorted(bar_dates, np.datetime64(day))) if self.tf != "D" else a["i"]
                if x >= len(f):
                    continue
                if a["type"] in ("buy", "add"):
                    out.append((x, "buy", "買" if a["type"] == "buy" else "加"))
                else:
                    ret = r["ret"] if a is r["actions"][-1] and r["end_i"] is not None else None
                    out.append((x, "sell", "賣" + (f" {ret:+.1%}" if ret is not None else "")))
        return out

    # ----- 繪圖 -----
    def redraw(self):
        self.glw.clear()
        self._keep = []           # 右軸 ViewBox 的參考，重畫時清掉
        if self.state is None or self.state.d.empty:
            return
        f = self._frame()
        dates = list(f["date"])
        live = self.bar is not None and self.tf == "D"
        if live:
            dates.append(self.bar["date"])
        x = np.arange(len(f))
        self._f, self._dates = f, dates
        plots = []

        price = self.glw.addPlot(row=0, col=0, axisItems={"bottom": DateAxis(dates)})
        price.showGrid(x=True, y=True, alpha=0.15)
        plots.append(price)
        price.addItem(CandleItem(x, f["open"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy(),
                                 f["close"].to_numpy()))
        if live:
            n = len(f)
            price.addItem(CandleItem([n], [self.bar["open"]], [self.bar["high"]], [self.bar["low"]],
                                     [self.bar["close"]], provisional=True))
        legend = price.addLegend(offset=(10, 5), labelTextSize="9pt", brush=pg.mkBrush(255, 255, 255, 200))
        for j, (p, name) in enumerate(MA_NAMES[self.tf].items()):
            if p in self.ma_on[self.tf] and f"ma{p}" in f:
                price.plot(x, f[f"ma{p}"].to_numpy(), pen=pg.mkPen(MA_COLORS[j], width=1.6), name=name, connect="finite")
        if self.bb_on:
            for col, style in (("bb_up", QtCore.Qt.DashLine), ("bb_mid", QtCore.Qt.DotLine), ("bb_low", QtCore.Qt.DashLine)):
                price.plot(x, f[col].to_numpy(), pen=pg.mkPen("#888", width=1, style=style),
                           name="布林通道" if col == "bb_up" else None, connect="finite")
        if self.tf == "D":
            for g in self.state.gaps:
                price.addItem(pg.InfiniteLine(g - 0.5, angle=90, pen=pg.mkPen(FLAT, style=QtCore.Qt.DotLine),
                                              label="資料缺口", labelOpts={"position": 0.95, "color": FLAT}))
        if self.buys_on or self.sells_on:
            hi, lo = f["high"].to_numpy(), f["low"].to_numpy()
            for xi, side, text in self._trade_marks(f):
                buy = side == "buy"
                if (buy and not self.buys_on) or (not buy and not self.sells_on):
                    continue
                t = pg.TextItem(text, color="w", fill=pg.mkBrush(UP if buy else DOWN), anchor=(0.5, 0 if buy else 1))
                t.setFont(QtGui.QFont("Microsoft JhengHei UI", 9, QtGui.QFont.Bold))
                t.setPos(xi, lo[xi] if buy else hi[xi])
                price.addItem(t)
        price.setMouseEnabled(x=True, y=False)

        row = 1
        if "rule" in self.panes_on:
            p = self._sub(row, dates, price, "RSI(3)")
            p.plot(x, f["rsi3"].to_numpy(), pen=pg.mkPen("#9b51e0", width=1.3), connect="finite")
            for lvl, col in ((25, UP), (70, DOWN)):
                p.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen(col, style=QtCore.Qt.DashLine),
                                          label=f"{'買' if lvl == 25 else '賣'} {lvl}",
                                          labelOpts={"position": 0.02, "color": col}))
            p.setYRange(0, 100)
            plots.append(p); row += 1
        if "vol" in self.panes_on:
            p = self._sub(row, dates, price, "成交量（張）")
            up = (f["close"] >= f["open"]).to_numpy()
            vol = f["volume"].to_numpy() / 1000
            p.addItem(pg.BarGraphItem(x=x[up], height=vol[up], width=0.7, brush=UP, pen=None))
            p.addItem(pg.BarGraphItem(x=x[~up], height=vol[~up], width=0.7, brush=DOWN, pen=None))
            plots.append(p); row += 1
        if "inst" in self.panes_on and self.tf == "D" and "foreign_net" in f:
            p = self._sub(row, dates, price, "三大法人（張）")
            p.addLegend(offset=(10, 5), labelTextSize="8pt", brush=pg.mkBrush(255, 255, 255, 200))
            w = 0.27
            for k, (col, name, color) in enumerate(INST):
                p.addItem(pg.BarGraphItem(x=x + (k - 1) * w, height=np.nan_to_num(f[col].to_numpy() / 1000),
                                          width=w, brush=color, pen=None, name=name))
            p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#aaa")))
            plots.append(p); row += 1
        if "margin" in self.panes_on and self.tf == "D" and "margin_bal" in f and f["margin_bal"].notna().any():
            p = self._sub(row, dates, price, "融資餘額（張）")
            p.addLegend(offset=(10, 5), labelTextSize="8pt", brush=pg.mkBrush(255, 255, 255, 200))
            p.plot(x, f["margin_bal"].to_numpy(), pen=pg.mkPen(UP, width=1.4), name="融資餘額（左軸）", connect="finite")
            right = pg.ViewBox()                          # 融券餘額用右邊的座標軸（數量差很多）
            p.showAxis("right")
            p.scene().addItem(right)
            p.getAxis("right").linkToView(right)
            p.getAxis("right").setLabel("融券（張）")
            right.setXLink(p)
            short = pg.PlotCurveItem(x, f["short_bal"].to_numpy(), pen=pg.mkPen(DOWN, width=1.4), connect="finite")
            right.addItem(short)
            p.legend.addItem(short, "融券餘額（右軸）")
            sb = f["short_bal"].to_numpy()
            if np.isfinite(sb).any():
                right.setYRange(np.nanmin(sb) * 0.9, np.nanmax(sb) * 1.1 + 1)

            def sync(*_, vb=p.getViewBox(), r=right):
                r.setGeometry(vb.sceneBoundingRect())
                r.linkedViewChanged(vb, r.XAxis)
            p.getViewBox().sigResized.connect(sync)
            sync()
            self._keep.append(right)
            plots.append(p); row += 1
        if "holders" in self.panes_on and self.tf == "D" and "pct_1000" in f and f["pct_1000"].notna().any():
            p = self._sub(row, dates, price, "大戶持股（%）")
            p.addLegend(offset=(10, 5), labelTextSize="8pt", brush=pg.mkBrush(255, 255, 255, 200))
            p.plot(x, f["pct_1000"].to_numpy(), pen=pg.mkPen("#c2410c", width=1.6), name="千張大戶", connect="finite")
            p.plot(x, f["pct_400"].to_numpy(), pen=pg.mkPen("#e0a100", width=1.2, style=QtCore.Qt.DashLine),
                   name="400 張以上", connect="finite")
            plots.append(p); row += 1
        if "macd" in self.panes_on:
            p = self._sub(row, dates, price, "MACD")
            p.addLegend(offset=(10, 5), labelTextSize="8pt", brush=pg.mkBrush(255, 255, 255, 200))
            h = np.nan_to_num(f["hist"].to_numpy())
            p.addItem(pg.BarGraphItem(x=x[h >= 0], height=h[h >= 0], width=0.6, brush=UP, pen=None))
            p.addItem(pg.BarGraphItem(x=x[h < 0], height=h[h < 0], width=0.6, brush=DOWN, pen=None))
            p.plot(x, f["dif"].to_numpy(), pen=pg.mkPen("#2f6fdb", width=1.2), name="DIF", connect="finite")
            p.plot(x, f["dea"].to_numpy(), pen=pg.mkPen("#e0a100", width=1.2), name="DEA", connect="finite")
            plots.append(p); row += 1
        if "kd" in self.panes_on:
            p = self._sub(row, dates, price, "KD(9)")
            p.addLegend(offset=(10, 5), labelTextSize="8pt", brush=pg.mkBrush(255, 255, 255, 200))
            p.plot(x, f["k"].to_numpy(), pen=pg.mkPen("#e0a100", width=1.3), name="K", connect="finite")
            p.plot(x, f["d"].to_numpy(), pen=pg.mkPen("#2f6fdb", width=1.3), name="D", connect="finite")
            for lvl in (20, 80):
                p.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen("#bbb", style=QtCore.Qt.DashLine)))
            p.setYRange(0, 100)
            plots.append(p); row += 1
        if "rsi" in self.panes_on:
            p = self._sub(row, dates, price, "RSI(14)")
            p.plot(x, f["rsi"].to_numpy(), pen=pg.mkPen("#0f9d8a", width=1.3), connect="finite")
            for lvl in (30, 70):
                p.addItem(pg.InfiniteLine(lvl, angle=0, pen=pg.mkPen("#bbb", style=QtCore.Qt.DashLine)))
            p.setYRange(0, 100)
            plots.append(p); row += 1

        self.glw.ci.layout.setRowStretchFactor(0, 5)
        for r in range(1, row):
            self.glw.ci.layout.setRowStretchFactor(r, 2)
        for p in plots[:-1]:
            p.getAxis("bottom").setStyle(showValues=False)
        last = len(dates) - 1
        view = {"D": VIEW_BARS, "W": 156, "M": 60}[self.tf]
        first = max(0, last - view)
        price.setXRange(first - 1, last + 2, padding=0)
        self._fit_y(price, f)
        price.getViewBox().sigXRangeChanged.connect(lambda *_: self._fit_y(price, f))
        self._vline = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#999", style=QtCore.Qt.DashLine)) for _ in plots]
        for p, ln in zip(plots, self._vline):
            p.addItem(ln, ignoreBounds=True)
        self._proxy = pg.SignalProxy(price.scene().sigMouseMoved, rateLimit=30, slot=lambda e: self._hover(e, price))
        self._show_info(min(last, len(f) - 1))

    def _sub(self, row, dates, link, label):
        p = self.glw.addPlot(row=row, col=0, axisItems={"bottom": DateAxis(dates)})
        p.setLabel("left", label)
        p.setXLink(link)
        p.setMouseEnabled(x=True, y=False)
        p.showGrid(x=True, y=False, alpha=0.1)
        p.getAxis("left").setWidth(60)
        link.getAxis("left").setWidth(60)
        return p

    def _fit_y(self, plot, f):
        (x0, x1), _ = plot.getViewBox().viewRange()
        seg = f.iloc[max(0, int(x0)):max(0, int(x1) + 1)]
        if not len(seg):
            return
        lo, hi = np.nanmin(seg["low"]), np.nanmax(seg["high"])
        if self.bb_on:
            lo, hi = min(lo, np.nanmin(seg["bb_low"])), max(hi, np.nanmax(seg["bb_up"]))
        if self.bar is not None and self.tf == "D" and x1 >= len(f):
            lo, hi = min(lo, self.bar["low"]), max(hi, self.bar["high"])
        plot.setYRange(lo * 0.97, hi * 1.03, padding=0)

    def _hover(self, evt, price):
        pos = evt[0]
        if not price.sceneBoundingRect().contains(pos):
            return
        i = int(round(price.getViewBox().mapSceneToView(pos).x()))
        if 0 <= i < len(self._f):
            for ln in self._vline:
                ln.setPos(i)
            self._show_info(i)

    def _show_info(self, i):
        f = self._f
        r = f.iloc[i]
        prev = f["close"].iat[i - 1] if i > 0 else np.nan
        chg = r["close"] / prev - 1 if prev else np.nan
        c = UP if chg > 0 else DOWN if chg < 0 else FLAT
        parts = [f"<b>{r['date']:%Y-%m-%d}</b>", f"開 {fmt(r['open'])}", f"高 {fmt(r['high'])}", f"低 {fmt(r['low'])}",
                 f"收 <b>{fmt(r['close'])}</b>", f"<span style='color:{c}'>{pct(chg)}</span>",
                 f"量 {fmt(r['volume'] / 1000, 0)} 張"]
        for j, (p, name) in enumerate(MA_NAMES[self.tf].items()):
            if p in self.ma_on[self.tf] and f"ma{p}" in f:
                parts.append(f"<span style='color:{MA_COLORS[j]}'>{name} {fmt(r[f'ma{p}'])}</span>")
        ind_parts = [f"RSI(3) {fmt(r['rsi3'], 0)}", f"RSI(14) {fmt(r['rsi'], 0)}", f"K {fmt(r['k'], 0)} D {fmt(r['d'], 0)}",
                     f"MACD 柱 {fmt(r['hist'])}"]
        if self.tf == "D" and "foreign_net" in f:
            ind_parts += [f"<span style='color:{col}'>{name} {fmt(r[k] / 1000, 0)} 張</span>" for k, name, col in INST]
        if self.tf == "D" and "margin_bal" in f:
            ind_parts += [f"融資 {fmt(r['margin_bal'], 0)} 張", f"融券 {fmt(r['short_bal'], 0)} 張"]
        if self.tf == "D" and "pct_1000" in f and pd.notna(r.get("pct_1000")):
            ind_parts.append(f"千張大戶 {fmt(r['pct_1000'])}%")
        self.info.setText("　".join(parts) + "<br>" + "　".join(ind_parts))


# ---------- 主視窗 ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model: app_model.Model):
        super().__init__()
        self.model = model
        self.pool = QtCore.QThreadPool.globalInstance()
        self.current: str | None = None
        self.summary = pd.DataFrame()
        self.intraday_cache: dict[str, dict] = {}
        self.setWindowTitle("台股分析工具")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self._build_toolbar()
        self._build_body()
        self.timer = QtCore.QTimer(self, interval=60_000, timeout=self._auto_quotes)
        self.timer.start()
        QtGui.QShortcut(QtGui.QKeySequence.Paste, self, activated=self.paste_screenshot)
        self.refresh_list()

    # ----- 版面 -----
    def _build_toolbar(self):
        tb = self.addToolBar("工具")
        tb.setMovable(False)
        self.act_quotes = tb.addAction("立即更新報價", self.update_quotes)
        self.act_auto = tb.addAction("盤中自動更新（每分鐘）")
        self.act_auto.setCheckable(True)
        self.act_auto.setChecked(True)
        tb.addSeparator()
        tb.addAction("加入股票代號", self.add_codes)
        tb.addAction("移除自選股", self.remove_code)
        tb.addSeparator()
        tb.addAction("匯入交易紀錄", self.import_trades)
        tb.addAction("下載交易紀錄範本", self.save_template)
        tb.addAction("加入截圖", self.add_screenshot_file)
        tb.addSeparator()
        tb.addAction("從 GitHub 更新資料", self.pull_data)
        tb.addAction("重新載入", self.reload)

    def _build_body(self):
        split = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        self.search = QtWidgets.QLineEdit(placeholderText="搜尋代號或名稱")
        self.search.textChanged.connect(self._filter)
        lv.addWidget(self.search)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicator(0, QtCore.Qt.AscendingOrder)   # 預設依代號，ETF 在最上面
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._selected)
        lv.addWidget(self.table)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.header = QtWidgets.QLabel("載入中…")
        self.header.setStyleSheet("font-size: 20px; font-weight: 600; padding: 4px;")
        rv.addWidget(self.header)
        self.advice = QtWidgets.QLabel()
        self.advice.setWordWrap(True)
        self.advice.setTextFormat(QtCore.Qt.RichText)
        self.advice.setStyleSheet("background: #f7f7f5; border: 1px solid #e4e3df; border-radius: 8px; padding: 10px;")
        self.advice.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        advice_scroll = QtWidgets.QScrollArea()          # 建議框可捲動、可拖曳分隔線調高度，讓圖表有空間
        advice_scroll.setWidget(self.advice)
        advice_scroll.setWidgetResizable(True)
        advice_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vsplit.addWidget(advice_scroll)
        self.tabs = QtWidgets.QTabWidget()
        self.chart = Chart()
        self.tabs.addTab(self.chart, "走勢圖")
        self.trades_tab = self._build_trades_tab()
        self.tabs.addTab(self.trades_tab, "交易紀錄")
        self.shots_tab = self._build_shots_tab()
        self.tabs.addTab(self.shots_tab, "截圖")
        vsplit.addWidget(self.tabs)
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 4)
        vsplit.setSizes([170, 700])
        rv.addWidget(vsplit, 1)
        split.addWidget(right)
        split.setSizes([430, 970])
        self.setCentralWidget(split)
        self.status = self.statusBar()

    def _build_trades_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("我的持股（依交易紀錄，先進先出，已扣手續費與證交稅）"))
        self.pos_table = self._simple_table(["代號", "股數", "平均成本", "現價", "未實現損益", "未實現 %", "已實現損益"])
        v.addWidget(self.pos_table, 1)
        v.addWidget(QtWidgets.QLabel("我的交易紀錄（跟規則比較：前後 3 天內規則有沒有同方向的買賣）"))
        self.my_table = self._simple_table(["日期", "代號", "買賣", "價格", "股數", "跟規則比較", "備註"])
        v.addWidget(self.my_table, 2)
        self.rule_label = QtWidgets.QLabel("規則在這檔的交易")
        v.addWidget(self.rule_label)
        self.rule_table = self._simple_table(["買進日", "買進價", "賣出日", "賣出價", "持有天數", "報酬（扣成本）", "賣出原因"])
        v.addWidget(self.rule_table, 2)
        return w

    def _build_shots_tab(self):
        w = QtWidgets.QSplitter()
        self.shot_list = QtWidgets.QListWidget()
        self.shot_list.currentItemChanged.connect(self._show_shot)
        w.addWidget(self.shot_list)
        self.shot_view = QtWidgets.QLabel("在視窗裡按 Ctrl+V 貼上截圖，或把圖片拖進來。")
        self.shot_view.setAlignment(QtCore.Qt.AlignCenter)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.shot_view)
        scroll.setWidgetResizable(True)
        w.addWidget(scroll)
        w.setSizes([260, 700])
        return w

    @staticmethod
    def _simple_table(headers):
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().hide()
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        return t

    @staticmethod
    def _fill(table, rows, colors=None):
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                it = QtWidgets.QTableWidgetItem(str(val))
                if colors and colors[r] and colors[r][c]:
                    it.setForeground(QtGui.QColor(colors[r][c]))
                table.setItem(r, c, it)

    # ----- 名單 -----
    def refresh_list(self):
        rng = self.model.date_range
        self.status.showMessage("計算名單中…" + (f"　資料 {rng[0]:%Y-%m-%d} ～ {rng[1]:%Y-%m-%d}" if rng else ""))
        job = Job(self.model.summary)
        job.signals.done.connect(self._list_ready)
        job.signals.failed.connect(self._error)
        self.pool.start(job)

    def _list_ready(self, summary: pd.DataFrame):
        self.summary = summary
        self._render_list()
        rng = self.model.date_range
        self.status.showMessage(f"資料 {rng[0]:%Y-%m-%d} ～ {rng[1]:%Y-%m-%d}，{len(summary)} 檔。"
                                f"規則：季線之上 RSI(3)<25 買；RSI(3)>70 或滿 20 天賣。" if rng else "沒有資料")
        if self.table.rowCount() and self.current is None:
            self.table.selectRow(0)
        if realtime.market_open() and self.act_auto.isChecked():
            self.update_quotes()

    def _render_list(self):
        sel = self.current
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.summary))
        q = self.model.quotes
        for r, row in enumerate(self.summary.itertuples()):
            live = not q.empty and row.code in q.index and pd.notna(q.at[row.code, "price"])
            price = q.at[row.code, "price"] if live else row.close
            chg = q.at[row.code, "change_pct"] if live else row.change_pct
            ic = self.intraday_cache.get(row.code)
            hint = "" if not ic else {"buy": "▲ 收現價會買", "sell": "▼ 收現價會賣"}.get(ic["action"], "—")
            cells = [row.code, row.name, fmt(price), pct(chg), row.action, hint]
            for c, val in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(val)
                if c == 2:
                    it.setData(QtCore.Qt.UserRole, float(price) if pd.notna(price) else 0.0)
                if c == 3 and pd.notna(chg):
                    it.setForeground(QtGui.QColor(UP if chg > 0 else DOWN if chg < 0 else FLAT))
                if c == 4:
                    it.setForeground(QtGui.QColor(ACTION_COLOR.get(row.action_code, FLAT)))
                if c == 5 and ic and ic["action"]:
                    it.setForeground(QtGui.QColor(UP if ic["action"] == "buy" else DOWN))
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        self._filter(self.search.text())
        if sel:
            self._select_code(sel)

    def _filter(self, text):
        text = text.strip().lower()
        for r in range(self.table.rowCount()):
            code = self.table.item(r, 0).text()
            name = self.table.item(r, 1).text().lower()
            self.table.setRowHidden(r, bool(text) and text not in code and text not in name)

    def _select_code(self, code):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == code:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                return

    def _selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self.current = self.table.item(rows[0].row(), 0).text()
        self.show_stock(self.current)

    # ----- 個股 -----
    def show_stock(self, code: str):
        s = self.model.analyze(code)
        q = self.model.quotes
        live = not q.empty and code in q.index and pd.notna(q.at[code, "price"])
        bar = None
        if live:
            qq = q.loc[code]
            label = "盤中" if realtime.market_open() else f"最新報價 {qq['date']:%m/%d}"
            price, chg, when = qq["price"], qq["change_pct"], f"{label} {qq['time']}"
            if len(s.d) and pd.notna(qq["date"]) and qq["date"] > s.d["date"].iat[-1]:
                bar = {"date": qq["date"], "open": qq["open"] if pd.notna(qq["open"]) else price,
                       "high": np.nanmax([qq["high"], price]), "low": np.nanmin([qq["low"], price]), "close": price}
        else:
            price = s.last_close
            prev = s.d["close"].iat[-2] if len(s.d) > 1 else np.nan
            chg, when = price / prev - 1 if prev else np.nan, f"收盤 {s.d['date'].iat[-1]:%Y-%m-%d}" if len(s.d) else ""
        color = UP if chg > 0 else DOWN if chg < 0 else FLAT
        self.header.setText(f"{s.name} {s.code}　<span style='color:{color}'>{fmt(price)}　{pct(chg)}</span>"
                            f"<span style='font-size:13px;color:#888'>　{when}</span>")
        a = s.action
        col = ACTION_COLOR.get(a["code"], FLAT)
        asof = f"依 {s.d['date'].iat[-1]:%Y-%m-%d} 收盤" if len(s.d) else "沒有資料"
        html = f"<span style='font-size:22px;font-weight:700;color:{col}'>{a['label']}</span>　" \
               f"<span style='color:#666'>（{asof}）</span><br>" + "<br>".join(a["lines"])
        chips = app_model.chip_lines(s.chips)
        if chips:
            html += "<hr><b>籌碼（三大法人、融資融券、集保大戶）</b><br>" + "<br>".join(chips)
        ic = self.intraday_cache.get(code)
        if ic:
            html += "<hr><b>盤中試算（如果今天收在現價）</b><br>" + "<br>".join(ic["lines"])
        self.advice.setText(html)
        self.chart.show_stock(s, bar)
        self._render_trades(s)
        self._render_shots(code)

    def _render_trades(self, s):
        pos = self.model.my_positions()
        self._fill(self.pos_table, [[r.code, f"{r.shares:,.0f}", fmt(r.avg_cost), fmt(r.last), f"{r.unrealized:,.0f}",
                                     pct(r.unrealized_pct), f"{r.realized:,.0f}"] for r in pos.itertuples()],
                   [[None, None, None, None, UP if r.unrealized > 0 else DOWN if r.unrealized < 0 else None, None,
                     UP if r.realized > 0 else DOWN if r.realized < 0 else None] for r in pos.itertuples()])
        t = self.model.my_trades()
        self._fill(self.my_table, [[f"{r.date:%Y-%m-%d}", r.code, "買" if r.side == "buy" else "賣", fmt(r.price),
                                    f"{r.shares:,.0f}", r.compare, r.note] for r in t.itertuples()])
        rows, colors = [], []
        dates = s.d["date"]
        reason = {"rsi3_high": "RSI 過熱", "max_days": "滿 20 天", "rsi3_low": ""}
        for r in reversed(s.sim["rounds"]):
            b = r["actions"][0]
            x = r["actions"][-1] if r["end_i"] is not None else None
            rows.append([f"{dates.iat[b['i']]:%Y-%m-%d}", fmt(b["px"]), f"{dates.iat[x['i']]:%Y-%m-%d}" if x else "持有中",
                         fmt(x["px"]) if x else fmt(s.last_close) + "（現價）",
                         (r["end_i"] if r["end_i"] is not None else len(dates) - 1) - r["start_i"], pct(r["ret"]),
                         reason.get(x["reason"], x["reason"]) if x else ""])
            colors.append([None] * 5 + [UP if r["ret"] > 0 else DOWN, None])
        self._fill(self.rule_table, rows, colors)
        done = [r["ret"] for r in s.sim["rounds"] if r["end_i"] is not None]
        if done:
            self.rule_label.setText(f"規則在這檔的交易：{len(done)} 筆，勝率 {np.mean(np.array(done) > 0):.0%}，"
                                    f"平均 {np.mean(done):+.2%}（已扣成本）")

    def _render_shots(self, code):
        self.shot_list.clear()
        for r in userdata.list_screenshots(code).iloc[::-1].itertuples():
            it = QtWidgets.QListWidgetItem(f"{r.date}　{r.note or Path(r.file).name}")
            it.setData(QtCore.Qt.UserRole, str(userdata.MYDATA / r.file))
            self.shot_list.addItem(it)
        if self.shot_list.count():
            self.shot_list.setCurrentRow(0)
        else:
            self.shot_view.setPixmap(QtGui.QPixmap())
            self.shot_view.setText("這檔還沒有截圖。在視窗裡按 Ctrl+V 貼上截圖，或把圖片拖進來。")

    def _show_shot(self, item, _prev=None):
        if not item:
            return
        pm = QtGui.QPixmap(item.data(QtCore.Qt.UserRole))
        if pm.isNull():
            self.shot_view.setText("讀不到這張圖")
        else:
            self.shot_view.setPixmap(pm.scaledToWidth(min(pm.width(), 1100), QtCore.Qt.SmoothTransformation))

    # ----- 盤中報價 -----
    def _auto_quotes(self):
        if self.act_auto.isChecked() and realtime.market_open():
            self.update_quotes()

    def update_quotes(self):
        self.act_quotes.setEnabled(False)
        self.status.showMessage("更新報價中…")
        job = Job(self._quotes_and_checks)
        job.signals.done.connect(self._quotes_ready)
        job.signals.failed.connect(self._error)
        self.pool.start(job)

    def _quotes_and_checks(self):
        q = self.model.fetch_quotes()
        checks = {}
        for code in q.index:
            try:
                checks[code] = self.model.intraday(code)
            except Exception:   # 單一檔試算失敗不影響其他檔
                pass
        return q, checks

    def _quotes_ready(self, result):
        q, checks = result
        self.act_quotes.setEnabled(True)
        self.intraday_cache = {k: v for k, v in checks.items() if v}
        self._render_list()
        if self.current:
            self.show_stock(self.current)
        errs = q.attrs.get("errors", [])
        when = q["time"].dropna().max() if len(q) else ""
        hits = [c for c, v in self.intraday_cache.items() if v["action"]]
        self.status.showMessage(f"報價時間 {when}（{dt.datetime.now(TZ):%H:%M:%S} 更新）"
                                + (f"　收在現價會出現訊號：{', '.join(hits)}" if hits else "")
                                + (f"　⚠ {len(errs)} 批失敗" if errs else ""))

    # ----- 使用者操作 -----
    def add_codes(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "加入股票代號", "代號（可一次輸入多個，用空白或逗號分開）：")
        if not ok or not text.strip():
            return
        codes = [c for c in text.replace(",", " ").replace("，", " ").split() if c]
        added, bad = self.model.add_codes(codes)
        if bad:
            QtWidgets.QMessageBox.warning(self, "找不到", f"資料裡找不到：{', '.join(bad)}")
        if added:
            self.current = added[0]
            self.refresh_list()

    def remove_code(self):
        if not self.current or self.current not in userdata.extra_codes():
            QtWidgets.QMessageBox.information(self, "移除自選股", "只能移除自己加入的代號（自選）。先在左邊選一檔。")
            return
        self.model.remove_code(self.current)
        self.current = None
        self.refresh_list()

    def import_trades(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "匯入交易紀錄", "", "CSV 或 Excel (*.csv *.xlsx *.xls)")
        if not path:
            return
        try:
            allt = userdata.import_trades(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "匯入失敗", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "匯入完成", f"目前共有 {len(allt)} 筆交易紀錄。")
        if self.current:
            self.show_stock(self.current)
        self.tabs.setCurrentWidget(self.trades_tab)

    def save_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存交易紀錄範本", "交易紀錄範本.csv", "CSV (*.csv)")
        if path:
            userdata.write_template(Path(path))
            QtWidgets.QMessageBox.information(self, "已儲存", "用 Excel 打開範本，照格式填好後再按「匯入交易紀錄」。")

    def add_screenshot_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "加入截圖", "", "圖片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self._save_shot(Path(path))

    def paste_screenshot(self):
        img = QtWidgets.QApplication.clipboard().image()
        if img.isNull():
            self.status.showMessage("剪貼簿裡沒有圖片（先用 Win+Shift+S 截圖）", 5000)
            return
        tmp = Path(tempfile.gettempdir()) / f"tw-shot-{dt.datetime.now():%H%M%S}.png"
        img.save(str(tmp))
        self._save_shot(tmp)

    def _save_shot(self, path: Path):
        code = self.current or ""
        note, ok = QtWidgets.QInputDialog.getText(self, "截圖備註", f"代號 {code or '（大盤）'}，備註（可留空）：")
        if not ok:
            return
        userdata.add_screenshot(path, code, dt.datetime.now(TZ).date(), note)
        self._render_shots(code)
        self.tabs.setCurrentWidget(self.shots_tab)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                self._save_shot(p)
            elif p.suffix.lower() in (".csv", ".xlsx", ".xls"):
                try:
                    userdata.import_trades(p)
                    self.status.showMessage(f"已匯入交易紀錄：{p.name}", 5000)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "匯入失敗", str(exc))
        if self.current:
            self.show_stock(self.current)

    def pull_data(self):
        out = app_model.git_pull()
        QtWidgets.QMessageBox.information(self, "從 GitHub 更新資料", out[-2000:])
        self.reload()

    def reload(self):
        self.status.showMessage("重新載入資料中…")
        job = Job(self.model.reload)
        job.signals.done.connect(lambda _: self.refresh_list())
        job.signals.failed.connect(self._error)
        self.pool.start(job)

    def _error(self, msg):
        self.act_quotes.setEnabled(True)
        self.status.showMessage("發生錯誤（詳細內容見對話框）")
        QtWidgets.QMessageBox.warning(self, "發生錯誤", msg[-3000:])


def _report_crash(exc_type, exc, tb) -> None:
    """未處理的錯誤：寫進 mydata/app_error.log 並跳出視窗（pythonw 沒有主控台，不這樣做會看不到錯誤）。"""
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        userdata.MYDATA.mkdir(parents=True, exist_ok=True)
        with open(userdata.MYDATA / "app_error.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n{text}")
    except OSError:
        pass
    if QtWidgets.QApplication.instance():
        QtWidgets.QMessageBox.critical(None, "台股分析工具發生錯誤",
                                       f"{text[-2500:]}\n\n（完整內容已記在 mydata/app_error.log）")


def main(argv: list[str] | None = None) -> int:
    sys.excepthook = _report_crash
    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("台股分析工具")
    font = app.font()
    font.setFamily("Microsoft JhengHei UI")
    font.setPointSize(10)
    app.setFont(font)
    splash = QtWidgets.QLabel("台股分析工具：載入資料中…")
    splash.setStyleSheet("font-size: 18px; padding: 30px; background: white;")
    splash.setWindowFlags(QtCore.Qt.SplashScreen)
    splash.show()
    app.processEvents()
    model = app_model.Model()
    win = MainWindow(model)
    splash.close()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
