@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  post_market.bat — Automatic Post-Market Data Pipeline
REM ═══════════════════════════════════════════════════════════════════════════
REM  Runs after market close (3:40 PM) to:
REM    1. Download today's 1-minute candles from Upstox Historical API
REM    2. Re-run the Feature Engine to update ML features
REM
REM  Called automatically by run_autostart.bat or scheduled via Task Scheduler.
REM  Requires: UPSTOX_ACCESS_TOKEN environment variable set
REM ═══════════════════════════════════════════════════════════════════════════

SET PROJECT_DIR=C:\Trading Platform\Advance Trading Model
cd /D "%PROJECT_DIR%"

echo.
echo ═══════════════════════════════════════════════════════════════
echo  POST-MARKET DATA PIPELINE — %DATE% %TIME%
echo ═══════════════════════════════════════════════════════════════

REM ── 1. Download today's data ──────────────────────────────────────────────
echo.
echo [1/2] Downloading latest historical data from Upstox...
echo      (1-minute candles → Renko bricks → Parquet files)
echo.
python main.py download
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Download encountered errors. Check logs.
) else (
    echo [OK] Download complete.
)

echo.
timeout /t 5 /nobreak >nul

REM ── 2. Re-generate features ──────────────────────────────────────────────
echo [2/2] Running Feature Engine on all data...
echo      (Computing velocity, wick_pressure, relative_strength, etc.)
echo.
python main.py features
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Feature engine encountered errors. Check logs.
) else (
    echo [OK] Features updated.
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo  POST-MARKET PIPELINE COMPLETE — %TIME%
echo  Data is ready for next trading day.
echo ═══════════════════════════════════════════════════════════════
echo.
timeout /t 10 /nobreak >nul
