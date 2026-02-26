@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  run_autostart.bat — Institutional Fortress Daily Launcher
REM ═══════════════════════════════════════════════════════════════════════════
REM
REM  Launches ALL 6 processes side by side:
REM    1. Live Engine        (signals + Mission Control feed)
REM    2. Paper Trader + API (FastAPI server + trading loop via server_main.py)
REM    3. Mission Control    (Streamlit :8501)
REM    4. Paper Dashboard    (Streamlit :8502)
REM    5. Post-Market        (auto at 15:40)
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
SET PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe

REM ── Verify Python exists ────────────────────────────────────────────────
if not exist "%PYTHON%" (
    echo [ERROR] .venv not found at %PROJECT_DIR%\.venv
    echo         Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo ═══════════════════════════════════════════════════════════════
echo  INSTITUTIONAL FORTRESS — %DATE% %TIME%
echo  Python: %PYTHON%
echo ═══════════════════════════════════════════════════════════════

REM ── 0. Wake UP Python Env ───────────────────────────────────────────────────
echo [%TIME%] Activating Virtual Environment...
cd /D "%PROJECT_DIR%"
call .venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate environment.
    pause
    exit /b 1
)

REM ── 1. Launch Live Engine (Signals + Virtual Execution) ─────────────────────
echo [1/3] Launching Live Engine ...
start "Fortress Engine" /D "%PROJECT_DIR%" cmd /k ""%PYTHON%" main.py live"

timeout /t 3 /nobreak >nul

REM ── 2. Launch API Server for Mobile App ───────────────────────────────────────
REM      Reads from live_state.json and serves it to Android over WebSocket
echo [2/3] Launching Mobile API Server (incl. News Engine) ...
start "Mobile API Server" /D "%PROJECT_DIR%" cmd /k ""%PYTHON%" server_main.py --api-only"

timeout /t 3 /nobreak >nul

REM ── 3. Schedule Post-Market Data Pipeline at 15:40 ────────────────────────────
echo [3/3] Scheduling post-market data download at 15:40 ...
schtasks /create /tn "Fortress_PostMarket" /tr "\"%PROJECT_DIR%\scripts\post_market.bat\"" /sc once /st 15:40 /f >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       Post-market pipeline scheduled for 15:40 today.
) else (
    echo       [NOTE] Could not auto-schedule. Run manually after 3:35 PM:
    echo              scripts\post_market.bat
)

echo.
echo  ┌──────────────────────────────────────────────────────────────┐
echo  │  MORNING SETUP COMPLETE. GOLDEN SETUP ACTIVE:                │
echo  │                                                              │
echo  │  1. Live Engine        = "Fortress Engine" window            │
echo  │       └─ Listens to WebSockets ^& executes virtual trades     │
echo  │  2. Mobile API Server  = "Mobile API Server" window          │
echo  │       └─ FastAPI server: http://0.0.0.0:8000                 │
echo  │       └─ WebSocket     : ws://0.0.0.0:8000/ws/telemetry        │
echo  │                                                              │
echo  │  Trading auto-shutdown : 3:35 PM (loop breaks)               │
echo  │  Data download starts  : 3:40 PM                             │
echo  └──────────────────────────────────────────────────────────────┘
echo.
timeout /t 10 /nobreak >nul
