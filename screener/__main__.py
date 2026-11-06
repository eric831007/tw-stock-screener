"""Command line entry point.

    python -m screener fetch                 # 抓今天（台灣時間）
    python -m screener fetch --date 2026-11-04
    python -m screener backfill --days 120   # 回補最近 120 個交易日
    python -m screener backfill --start 2004-01-01 --end 2004-12-31  # 回補指定期間
    python -m screener status                # 看目前存了多少資料
    python -m screener dashboard             # 產生權值股技術分析儀表板
    python -m screener optimize              # 每支股票各自找最高勝率規則並驗證（寫到 reports/）
    python -m screener app                   # 開啟桌面分析工具（需要 PySide6、pyqtgraph）
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from . import fetch as fetch_mod
from . import store
from .fetch import backfill, backfill_range, fetch_day, today_taipei, update_stock_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="screener", description="台股熱門標的分析工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="抓取單一交易日")
    p_fetch.add_argument("--date", type=dt.date.fromisoformat, help="YYYY-MM-DD，預設今天")

    p_back = sub.add_parser("backfill", help="回補過去 N 個交易日")
    p_back.add_argument("--days", type=int, default=120)
    p_back.add_argument("--start", type=dt.date.fromisoformat, help="YYYY-MM-DD，與 --end 一起用")
    p_back.add_argument("--end", type=dt.date.fromisoformat, help="YYYY-MM-DD")

    sub.add_parser("status", help="顯示已存資料概況")
    sub.add_parser("dashboard", help="產生權值股技術分析儀表板 dashboard/index.html")
    sub.add_parser("app", help="開啟桌面分析工具")
    p_opt = sub.add_parser("optimize", help="每支股票各自找最高勝率規則，並用沒看過的期間驗證")
    p_opt.add_argument("--split", type=dt.date.fromisoformat, default=dt.date(2025, 1, 1),
                       help="切分日 YYYY-MM-DD：之前用來挑規則、之後用來驗證（預設 2025-01-01）")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.command == "fetch":
        day = args.date or today_taipei()
        update_stock_info()
        status = fetch_day(day)
        fetch_mod.catch_up_optional()     # 融資融券若比排程晚公布，下次執行補上
        fetch_mod.update_holders()        # 集保股權分散表（每週）
        print(f"{day}：{status}")
        return _report_failures() or (1 if status == "partial" else 0)

    if args.command == "backfill":
        update_stock_info()
        if args.start:
            end = min(args.end or today_taipei(), today_taipei())
            backfill_range(args.start, end)
        else:
            backfill(args.days)
        return _report_failures()

    if args.command == "dashboard":
        from .dashboard import build
        print(f"已產生 {build()}")
        return 0

    if args.command == "app":
        from .desktop import main as app_main
        return app_main([sys.argv[0]])

    if args.command == "optimize":
        from .optimize import run
        csv_path, md_path = run(args.split)
        print(f"已產生 {csv_path}")
        print(f"已產生 {md_path}")
        return 0

    if args.command == "status":
        days = store.trading_days()
        if not days:
            print("尚未存任何資料，先執行 backfill。")
            return 0
        incomplete = [d for d in days if not store.is_complete(d)]
        print(f"交易日：{len(days)} 天（{days[0]} ～ {days[-1]}）")
        print(f"資料不完整的日子：{len(incomplete)} 天 {incomplete[-5:] if incomplete else ''}")
        print(f"產業別資料：{len(store.load_stock_info())} 檔")
        return 0
    return 0


def _report_failures() -> int:
    if not fetch_mod.failures:
        return 0
    print(f"\n有 {len(fetch_mod.failures)} 項抓取失敗（下次執行會自動補抓）：")
    for f in fetch_mod.failures:
        print("  -", f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
