"""確認命令列入口能執行（之前 __main__.py 有語法錯誤，其他測試都沒抓到）。"""
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_all_modules_compile():
    assert compileall.compile_dir(ROOT / "screener", quiet=1, force=True)


def test_cli_help_lists_commands():
    r = subprocess.run([sys.executable, "-m", "screener", "--help"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    for cmd in ("fetch", "backfill", "status", "dashboard", "optimize", "app"):
        assert cmd in r.stdout
