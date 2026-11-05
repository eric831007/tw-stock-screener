# tw-stock-screener 台股熱門標的分析工具

每個交易日收盤後，自動抓取台股上市櫃行情與三大法人買賣超，累積成歷史資料，
之後依「量、價、法人籌碼、技術指標」算出熱度分數並排行。

> 熱度高代表「有資金在買」，不代表一定會漲。本工具只用來縮小研究範圍，不構成投資建議。

## 目前進度

| 週次 | 內容 | 狀態 |
| --- | --- | --- |
| 第 1 週 | 資料抓取、存檔、GitHub Actions 每日自動執行 | ✅ 完成 |
| 第 2 週 | 回補 120 個交易日、指標計算 | 未開始 |
| 第 3 週 | 熱度評分、Excel 報表、Email 寄送 | 未開始 |
| 第 5–6 週 | 回測 | 未開始 |

## 資料來源

| 資料 | 上市 | 上櫃 |
| --- | --- | --- |
| 每日行情 | 證交所 MI_INDEX | 櫃買中心 dailyQuotes |
| 三大法人 | 證交所 T86 | 櫃買中心 insti/dailyTrade |
| 產業別 | 證交所 OpenAPI t187ap03_L | 櫃買中心 OpenAPI mopsfin_t187ap03_O |

每次請求間隔 3 秒，避免被交易所封鎖。

## 資料存放

```
data/prices/2026/2026-11-04_TWSE.csv        每日行情（成交量單位：股）
data/institutional/2026/2026-11-04_TPEx.csv 三大法人買賣超（單位：股）
data/holidays.csv                           已確認的休市日
data/stock_info.csv                         代號、名稱、產業別
```

## 在自己電腦上執行

```bash
pip install -r requirements.txt
python -m screener backfill --days 120   # 第一次：回補最近 120 個交易日（約 30–40 分鐘）
python -m screener fetch                 # 之後每天：抓今天
python -m screener status                # 看存了多少資料
pytest                                   # 跑測試
```

## GitHub Actions 自動執行

`.github/workflows/daily.yml` 會在週一到週五台灣時間 16:40 自動執行，
抓完的資料會自動 commit 回這個 repo。

第一次使用：到 repo 的 **Actions** 分頁 → 選「每日抓取台股資料」→ **Run workflow**，
`backfill_days` 填 `120`，就會回補歷史資料。

## 回補很久以前的歷史資料

`.github/workflows/history.yml`：到 **Actions** → 「回補歷史資料（依年份）」→ **Run workflow**，
填起訖年份（最早 2004）。每一年依序跑一次，打包成 `history-YYYY.zip` 放到 repo 的 **Releases → history**，
不放進 `data` 資料夾，以免 repo 太大。

資料最早大約到：上市行情 2004 年、上櫃行情 2007 年、上市三大法人 2012 年；更早的年份只會有部分資料。

## 評分設定

權重、流動性門檻等都在 `config.yaml`，改那裡就好，不需要改程式。
