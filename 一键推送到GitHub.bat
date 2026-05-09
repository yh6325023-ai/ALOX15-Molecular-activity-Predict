@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo === 推送到 GitHub（会用本仓库里配置的代理，默认 127.0.0.1:7890）===
echo === 请先打开 Clash / VPN，确保 Chrome 能打开 github.com ===
echo.
git push --force-with-lease origin main
echo.
if errorlevel 1 (
  echo 推送失败：检查代理是否已开、端口是否为 7890（不是则改 .git\config 里 http.proxy）
) else (
  echo 成功：已推送到 GitHub（origin/main）。
)
echo.
pause
