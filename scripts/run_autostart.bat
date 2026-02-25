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

REM ── 1. Launch Live Engine (signals + Mission Control feed) ───────────────
echo [1/5] Launching Live Engine ...
start "Fortress Engine" /D "%PROJECT_DIR%" cmd /k ""%PYTHON%" main.py live"

timeout /t 3 /nobreak >nul

REM ── 2. Launch Paper Trader + FastAPI Server (bundled) ───────────────────
REM      server_main.py runs BOTH concurrently:
REM        • Paper trading loop  (asyncio.to_thread → ThreadPoolExecutor)
REM        • FastAPI + WebSocket (uvicorn on 0.0.0.0:8000)
REM      Android app connects to:
REM        WebSocket:  ws://<tailscale-ip>:8000/ws/telemetry
REM        Commands:   POST http://<tailscale-ip>:8000/api/command
echo [2/5] Launching Paper Trader + FastAPI Server (server_main.py) ...
start "Paper Trader + API" /D "%PROJECT_DIR%" cmd /k ""%PYTHON%" server_main.py"

timeout /t 3 /nobreak >nul

REM ── 3. Launch Mission Control Dashboard ──────────────────────────────────
echo [3/5] Launching Mission Control dashboard ...
start "Mission Control" /D "%PROJECT_DIR%" cmd /k ""%PYTHON%" -m streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true"

timeout /t 2 /nobreak >nul

REM ── 4. Launch Paper Trading Dashboard ────────────────────────────────────
echo [4/5] Launching Paper Trading dashboard ...
start "Paper Dashboard" /D "%PROJECT_DIR%" cmd /k ""%PYTHON%" -m streamlit run src/ui/paper_dashboard.py --server.port 8502 --server.headless true"

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
echo  ┌──────────────────────────────────────────────────────────────┐
echo  │  ALL PROCESSES RUNNING:                                      │
echo  │                                                              │
echo  │  1. Live Engine        = "Fortress Engine" window            │
echo  │  2. Paper Trader + API = "Paper Trader + API" window         │
echo  │       └─ Trading loop  : running inside server_main.py       │
echo  │       └─ FastAPI server: http://localhost:8000               │
echo  │       └─ WebSocket     : ws://localhost:8000/ws/telemetry    │
echo  │       └─ Android cmd   : POST /api/command                   │
echo  │  3. Mission Control    = http://localhost:8501               │
echo  │  4. Paper Dashboard    = http://localhost:8502               │
echo  │  5. Post-Market        = Auto at 15:40 (download+feat)       │
echo  │                                                              │
echo  │  Trading auto-shutdown : 3:35 PM (loop breaks)               │
echo  │  Data download starts  : 3:40 PM                             │
echo  └──────────────────────────────────────────────────────────────┘
echo.
timeout /t 10 /nobreak >nul
