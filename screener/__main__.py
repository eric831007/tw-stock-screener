"""Command line entry point.

    python -m screener fetch                 # 抓今天（台灣時間）
    python -m screener fetch --date 2026-11-04
    python -m screener backfill --days 120   # 回補最近 120 個交易日
    python -m screener backfill --start 2004-01-01 --end 2004-12-31  # 回補指定期間
    python -m screener status                # 看目前存了多少資料
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.command == "fetch":
        day = args.date or today_taipei()
        update_stock_info()
        status = fetch_day(day)
        print(f"{day}：{status}")
        return 1 if status == "partial" else 0

    if args.command == "backfill":
        update_stock_info()
        if args.start:
            end = min(args.end or today_taipei(), today_taipei())
            backfill_range(args.start, end)
        else:
            backfill(args.days)
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


if __name__ == "__main__":
    sys.exit(main())
