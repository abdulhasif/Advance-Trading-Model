@echo off
REM ----------------------------------------------------------------------------
REM  Setup Institutional Fortress Daily Task (Run as Administrator)
REM ----------------------------------------------------------------------------

echo.
echo  Installing 'Institutional Fortress' startup task...
echo.

REM Create the task: Daily at 08:50 AM, Admin rights, Pointing to run_autostart.bat
REM IMPORTANT: We must use the FULL ABSOLUTE PATH to run_autostart.bat

schtasks /create /tn "Institutional Fortress" /tr "\"C:\Trading Platform\Advance Trading Model\scripts\run_autostart.bat\"" /sc daily /st 08:50 /rl HIGHEST /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  ----------------------------------------------------------------------
    echo  SUCCESS: Task created!
    echo  The system will now start automatically at 08:50 AM every day.
    echo  ----------------------------------------------------------------------
) else (
    echo.
    echo  ----------------------------------------------------------------------
    echo  ERROR: Could not create task.
    echo  PLEASE RIGHT-CLICK THIS FILE AND SELECT "RUN AS ADMINISTRATOR"
    echo  ----------------------------------------------------------------------
)

echo.
pause
