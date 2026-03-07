while ($true) {
    if (Test-Path "storage\logs\download_new.log") {
        $content = Get-Content "storage\logs\download_new.log" -Tail 5 -ErrorAction SilentlyContinue
        if ($content -match "DONE") {
            break
        }
    }
    Start-Sleep -Seconds 30
}
Write-Host "Download complete. Starting feature engine..."
.venv\Scripts\python.exe main.py features 2>&1 | Tee-Object -FilePath storage\logs\features_new.log

Write-Host "Features complete. Starting model training..."
.venv\Scripts\python.exe main.py train 2>&1 | Tee-Object -FilePath storage\logs\train_new.log

Write-Host "Pipeline completely finished!"
