param (
    [switch]$Verbose = $false,   # Default Quiet, se passato diventa Verbose
    [switch]$DryRun = $false  # Se passato, ferma lo script dopo la Build e non cancella i file
)

$script:needsReset = $false
$script:success = $false

# Definiamo i flag basandoci sull'inverso di $Verbose
$uvQuietFlag = if ($Verbose) { "" } else { "-q" }
$svnQuietFlag = if ($Verbose) { "" } else { "-q" }

function Check-LastCommand {
    param([string]$message)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRORE: $message" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

try {
    # Step 1: Lettura versione short
    if ($Verbose) { Write-Host "Recupero versione attuale..." -ForegroundColor Gray }
    $versionFullRaw = uv version --short
    Check-LastCommand "Lettura versione fallita."
    $versionParts = $versionFullRaw.Split('.')
    $shortVersion = "$($versionParts[0]).$($versionParts[1])"

    # Step 2: Generazione versione FULL tramite SVN
    if ($Verbose) { Write-Host "Aggiornamento SVN..." -ForegroundColor Gray }
    svn update $svnQuietFlag
    $revision = (svn info --show-item last-changed-revision).Trim()
    Check-LastCommand "Errore SVN."

    $fullVersion = "$shortVersion.$revision"

    # Step 3: Assegnazione versione FULL
    Write-Host "Impostazione versione FULL: $fullVersion ..." -ForegroundColor Cyan
    uv version $uvQuietFlag $fullVersion
    Check-LastCommand "Errore setting versione full."
    $script:needsReset = $true
    Write-Host "Versione impostata`n" -ForegroundColor Green

    # Step 4: Build
    Write-Host "Esecuzione Build..." -ForegroundColor Cyan
    uv build $uvQuietFlag
    Check-LastCommand "Errore build."
    Write-Host "Build completata`n" -ForegroundColor Green

    # Controllo per DryRun
    if ($DryRun) {
        Write-Host "Modalità DryRun attiva. I file sono disponibili nella cartella 'dist'.`n" -ForegroundColor Yellow
        $script:success = $true
        exit 0
    }

    # Step 5: Credenziali e Pubblicazione
    Write-Host "Pubblicazione in corso..." -ForegroundColor Cyan

    $gbsUser = Read-Host "Inserisci lo username (codice GBS)"
    $securePass = Read-Host "Inserisci la password" -AsSecureString
    $plainPass = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePass))

    uv publish $uvQuietFlag -u $gbsUser -p $plainPass
    Check-LastCommand "Errore pubblicazione."
    Write-Host "Pubblicazione completata con successo`n" -ForegroundColor Green

    $script:success = $true
}
catch {
    Write-Host "Errore imprevisto: $_" -ForegroundColor Red
}
finally {
    # 1. Reset automatico della versione (Sempre, se è stata cambiata)
    if ($script:needsReset) {
        if ($Verbose) { Write-Host "[PULIZIA] Ripristino versione originale ($shortVersion)..." -ForegroundColor Yellow }
        uv version $uvQuietFlag $shortVersion | Out-Null
    }

    # 2. Rimozione cartelle dist e build (Solo se NON è DryRun)
    if (-not $DryRun) {
        foreach ($folder in "dist", "build") {
            if (Test-Path $folder) {
                if ($Verbose) { Write-Host "[PULIZIA] Rimozione cartella $folder..." -ForegroundColor Yellow }
                Remove-Item -Path $folder -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # 3. Messaggio finale
    if ($script:success) {
        Write-Host "===============================================" -ForegroundColor Green
        $msg = if ($DryRun) { "  BUILD COMPLETATA CON SUCCESSO!" } else { "  OPERAZIONE COMPLETATA CON SUCCESSO!" }
        Write-Host $msg -ForegroundColor Green
        Write-Host "===============================================" -ForegroundColor Green
    }
}