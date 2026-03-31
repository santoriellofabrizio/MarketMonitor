@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo MarketMonitor - Build EXE
echo ================================================================================
echo.
echo Produce tre eseguibili:
echo   run-strategy.exe      (ONEDIR, avviato da Control Panel o da CLI)
echo   run-control-panel.exe (ONEFILE, lancia run-strategy via QProcess)
echo   run-dashboard.exe     (ONEFILE, viewer dati standalone)
echo.

:: ============================================================================
:: Configurazione
:: ============================================================================
set SPEC_STRATEGY=run_strategy.spec
set SPEC_PANEL=run_control_panel.spec
set SPEC_DASHBOARD=run_dashboard.spec
set SPEC_FILE=%SPEC_STRATEGY%
set OUTPUT_DIR=dist\run-strategy
set DEPLOY_DIR=dist\MarketMonitor_Deploy

:: ============================================================================
:: Verifica prerequisiti
:: ============================================================================
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] uv non trovato. Installa da https://docs.astral.sh/uv/
    exit /b 1
)

if not exist "%SPEC_STRATEGY%" (
    echo [ERRORE] File spec non trovato: %SPEC_STRATEGY%
    exit /b 1
)
if not exist "%SPEC_PANEL%" (
    echo [ERRORE] File spec non trovato: %SPEC_PANEL%
    exit /b 1
)

:: ============================================================================
:: Sincronizzazione dipendenze (garantisce sfm_datalibrary e tutte le lib nel venv)
:: ============================================================================
echo [0/3] Sincronizzazione dipendenze con uv sync...
echo.
uv sync
if errorlevel 1 (
    echo.
    echo [ERRORE] uv sync fallito. Verifica accesso al NAS e al lockfile.
    exit /b 1
)

:: ============================================================================
:: Build con PyInstaller (tramite uv run: usa il venv corretto)
:: ============================================================================
echo.
echo [1/4] Build run-strategy (ONEDIR)...
echo.
uv run pyinstaller %SPEC_STRATEGY% --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERRORE] Build run-strategy fallita. Controlla l'output sopra.
    exit /b 1
)
echo   OK: %OUTPUT_DIR%\

echo.
echo [2/4] Build run-control-panel (ONEFILE)...
echo.
uv run pyinstaller %SPEC_PANEL% --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERRORE] Build run-control-panel fallita. Controlla l'output sopra.
    exit /b 1
)
echo   OK: dist\run-control-panel.exe

echo.
echo [3/4] Build run-dashboard (ONEFILE)...
echo.
if exist "%SPEC_DASHBOARD%" (
    uv run pyinstaller %SPEC_DASHBOARD% --clean --noconfirm
    if errorlevel 1 (
        echo   WARN: Build run-dashboard fallita. Continuazione...
    ) else (
        echo   OK: dist\run-dashboard.exe
    )
) else (
    echo   WARN: %SPEC_DASHBOARD% non trovato, skip.
)

:: ============================================================================
:: Crea cartella deployment pronta all'uso
:: ============================================================================
echo.
echo [4/4] Preparazione cartella deployment: %DEPLOY_DIR%\

if exist "%DEPLOY_DIR%" rmdir /s /q "%DEPLOY_DIR%"
mkdir "%DEPLOY_DIR%"

:: run-strategy: exe + _internal\ (ONEDIR)
xcopy /e /i /q "%OUTPUT_DIR%\*" "%DEPLOY_DIR%\" >nul
echo   - Copiato: run-strategy.exe + _internal\

:: run-control-panel: singolo exe (ONEFILE)
if exist "dist\run-control-panel.exe" (
    copy /y "dist\run-control-panel.exe" "%DEPLOY_DIR%\run-control-panel.exe" >nul
    echo   - Copiato: run-control-panel.exe
)

:: run-dashboard: singolo exe (ONEFILE)
if exist "dist\run-dashboard.exe" (
    copy /y "dist\run-dashboard.exe" "%DEPLOY_DIR%\run-dashboard.exe" >nul
    echo   - Copiato: run-dashboard.exe
)

:: etc/config
if exist "etc\config" (
    xcopy /e /i /q "etc\config" "%DEPLOY_DIR%\etc\config\" >nul
    echo   - Copiato: etc\config\
    echo     ATTENZIONE: aggiornare i path assoluti nei file YAML!
) else (
    echo   - WARN: etc\config\ non trovata, skip.
)

:: user_strategy NON copiata automaticamente
echo   - user_strategy\: NON copiata automaticamente.
echo     Copia a mano solo le strategie necessarie in %DEPLOY_DIR%\user_strategy\

echo.
echo ================================================================================
echo Build completata con successo!
echo ================================================================================
echo.
echo STRUTTURA DEPLOYMENT (%DEPLOY_DIR%\):
echo   run-strategy.exe          ^<-- avviato da Control Panel o da CLI
echo   run-control-panel.exe     ^<-- avvia le strategie (lancia run-strategy)
echo   run-dashboard.exe         ^<-- viewer dati standalone
echo   _internal\                ^<-- librerie Python di run-strategy (non toccare)
echo   user_strategy\            ^<-- COPIA A MANO le strategie che ti servono
echo   etc\config\               ^<-- aggiornare i path nei file YAML
echo.
echo AVVIO TRAMITE CONTROL PANEL (consigliato):
echo   run-control-panel.exe
echo   run-control-panel.exe --config ^<nome_config^>
echo.
echo AVVIO DIRETTO DA CLI:
echo   run-strategy.exe ^<nome_config^>
echo   run-strategy.exe --list
echo ================================================================================

endlocal
