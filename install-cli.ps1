# install-cli.ps1 - Aggiunge i comandi MarketMonitor al PATH utente

$binPath = "C:\AFMachineLearning\Projects\Trading\MarketMonitorFI\bin"

# Ottieni PATH utente attuale
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*$binPath*") {
    $newPath = "$userPath;$binPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "[OK] Aggiunto $binPath al PATH utente" -ForegroundColor Green
    Write-Host "    Riapri il terminale per usare i comandi:" -ForegroundColor Yellow
    Write-Host "    - run-strategy" -ForegroundColor Cyan
    Write-Host "    - run-dashboard" -ForegroundColor Cyan
    Write-Host "    - run-mock" -ForegroundColor Cyan
} else {
    Write-Host "[INFO] $binPath e' gia' nel PATH" -ForegroundColor Yellow
}
