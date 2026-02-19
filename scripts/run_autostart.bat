@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  run_autostart.bat — Institutional Fortress Daily Launcher
REM ═══════════════════════════════════════════════════════════════════════════
REM
REM  Launches BOTH the live engine and paper trader side by side,
REM  plus both dashboards. This lets you watch live signals on Mission
REM  Control while the paper trader validates trades with virtual capital.
REM
REM  HOW TO ADD TO WINDOWS TASK SCHEDULER (daily at 08:50 AM):
REM  ──────────────────────────────────────────────────────────
REM    1. Win+R → taskschd.msc → Enter
REM    2. "Create Basic Task..."
REM    3. Name: "Institutional Fortress" | Trigger: Daily 08:50:00
REM    4. Action → Start a program:
REM         Program:   C:\Trading Platform\Advance Trading Model\scripts\run_autostart.bat
REM         Start in:  C:\Trading Platform\Advance Trading Model
REM    5. Properties → "Run only when user is logged on" + "Highest privileges"
REM    6. Conditions → Uncheck "Start only if on AC power"
REM
REM ═══════════════════════════════════════════════════════════════════════════

SET PROJECT_DIR=C:\Trading Platform\Advance Trading Model

REM ── Activate environment (edit for conda/venv) ──────────────────────────
REM call conda activate fortress_env
REM call "%PROJECT_DIR%\venv\Scripts\activate.bat"
SET ACTIVATE_CMD=echo Using system Python
%ACTIVATE_CMD%

echo ═══════════════════════════════════════════════════════════════
echo  INSTITUTIONAL FORTRESS — %DATE% %TIME%
echo ═══════════════════════════════════════════════════════════════

REM ── 1. Launch Live Engine (signals + Mission Control feed) ───────────────
echo [1/4] Launching Live Engine ...
start "Fortress Engine" /D "%PROJECT_DIR%" cmd /k "python main.py live"

timeout /t 3 /nobreak >nul

REM ── 2. Launch Paper Trader (virtual trades) ──────────────────────────────
echo [2/4] Launching Paper Trading Engine ...
start "Paper Trader" /D "%PROJECT_DIR%" cmd /k "python main.py paper"

timeout /t 3 /nobreak >nul

REM ── 3. Launch Mission Control Dashboard ──────────────────────────────────
echo [3/4] Launching Mission Control dashboard ...
start "Mission Control" /D "%PROJECT_DIR%" cmd /k "python -m streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true"

timeout /t 2 /nobreak >nul

REM ── 4. Launch Paper Trading Dashboard ────────────────────────────────────
echo [4/5] Launching Paper Trading dashboard ...
start "Paper Dashboard" /D "%PROJECT_DIR%" cmd /k "python -m streamlit run src/ui/paper_dashboard.py --server.port 8502 --server.headless true"

timeout /t 2 /nobreak >nul

REM ── 5. Schedule Post-Market Data Pipeline at 15:40 ─────────────────────
echo [5/5] Scheduling post-market data download at 15:40 ...
schtasks /create /tn "Fortress_PostMarket" /tr "\"%PROJECT_DIR%\scripts\post_market.bat\"" /sc once /st 15:40 /f >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       Post-market pipeline scheduled for 15:40 today.
) else (
    echo       [NOTE] Could not auto-schedule. Run manually after 3:35 PM:
    echo              scripts\post_market.bat
)

echo.
echo  ┌──────────────────────────────────────────────────────┐
echo  │  RUNNING 5 PROCESSES:                                │
echo  │                                                      │
echo  │  1. Live Engine      = "Fortress Engine" window      │
echo  │  2. Paper Trader     = "Paper Trader" window         │
echo  │  3. Mission Control  = http://localhost:8501          │
echo  │  4. Paper Dashboard  = http://localhost:8502          │
echo  │  5. Post-Market      = Auto at 15:40 (download+feat) │
echo  │                                                      │
echo  │  Trading auto-shutdown at 3:35 PM                    │
echo  │  Data download starts at 3:40 PM                     │
echo  └──────────────────────────────────────────────────────┘
echo.
timeout /t 10 /nobreak >nul
