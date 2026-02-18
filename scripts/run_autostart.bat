@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  run_autostart.bat — Institutional Fortress Daily Launcher
REM ═══════════════════════════════════════════════════════════════════════════
REM
REM  HOW TO ADD TO WINDOWS TASK SCHEDULER (daily at 08:50 AM):
REM  ──────────────────────────────────────────────────────────
REM    1. Win+R → taskschd.msc → Enter
REM    2. "Create Basic Task..."
REM    3. Name: "Institutional Fortress" | Trigger: Daily 08:50:00
REM    4. Action → Start a program:
REM         Program:   C:\Trading Platform\Advance Trading Model\run_autostart.bat
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

REM ── Launch Live Engine ──────────────────────────────────────────────────
echo [1/2] Launching Live Engine ...
start "Fortress Engine" /D "%PROJECT_DIR%" cmd /k "python main.py live"

timeout /t 5 /nobreak >nul

REM ── Launch Dashboard ────────────────────────────────────────────────────
echo [2/2] Launching Dashboard ...
start "Fortress Dashboard" /D "%PROJECT_DIR%" cmd /k "python main.py dashboard"

echo.
echo  Engine:    see "Fortress Engine" window
echo  Dashboard: http://localhost:8501
echo.
timeout /t 10 /nobreak >nul
