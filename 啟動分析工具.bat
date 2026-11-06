@echo off
chcp 65001 >nul
rem 台股分析工具：點兩下開啟
cd /d "%~dp0"
if not exist ".venv\pythonw.exe" (
  echo 找不到 .venv\pythonw.exe，請先建立 Python 3.12 環境（見 README 的「桌面分析工具」）。
  pause
  exit /b 1
)
start "" ".venv\pythonw.exe" -m screener app
