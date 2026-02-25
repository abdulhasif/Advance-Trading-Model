# install_task.ps1 — One-click setup for Windows 11 Task Scheduler
# Run this script as Administrator to schedule the trading system daily at 08:50 AM.

$TaskName = "Institutional Fortress"
$ProjectDir = "C:\Trading Platform\Advance Trading Model"
$ScriptPath = "$ProjectDir\scripts\run_autostart.bat"
$Time = "08:50am"

Write-Host "Installing '$TaskName' to run daily at $Time..." -ForegroundColor Cyan

# 1. Define Action (Launch the batch file)
$Action = New-ScheduledTaskAction -Execute $ScriptPath -WorkingDirectory $ProjectDir

# 2. Define Trigger (Daily at 08:50)
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

# 3. Define Settings (Wake computer, run with highest privileges)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -WakeToRun

# 4. Register the Task
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

try {
    # Unregister existing if any
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    
    # Register new
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force -ErrorAction Stop
    
    Write-Host "`n✅ SUCCESS: Task '$TaskName' has been scheduled." -ForegroundColor Green
    Write-Host "   - Time: Daily at $Time"
    Write-Host "   - Script: $ScriptPath"
    Write-Host "   - Mode: $env:USERNAME (Highest Privileges)"
    Write-Host "`nTo test it now, open Start -> Task Scheduler, find '$TaskName', right-click -> Run." -ForegroundColor Gray
} catch {
    Write-Host "`n❌ FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "👉 You must run this script as Administrator." -ForegroundColor Yellow
}

Write-Host "`n"
Read-Host "Press Enter to close..."
