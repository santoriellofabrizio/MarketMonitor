@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo MarketMonitor - Build EXE (Approccio Ibrido)
echo ================================================================================
echo.
echo Il bundle contiene SOLO il framework core (market_monitor).
echo Le strategie (user_strategy/) restano file .py esterni, NON nel bundle.
echo.

:: ============================================================================
:: Configurazione
:: ============================================================================
set SPEC_FILE=run_strategy.spec
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

if not exist "%SPEC_FILE%" (
    echo [ERRORE] File spec non trovato: %SPEC_FILE%
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
echo [1/3] Avvio build PyInstaller...
echo.
uv run pyinstaller %SPEC_FILE% --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [ERRORE] Build fallita. Controlla l'output sopra.
    exit /b 1
)

echo.
echo [2/3] Build completata: %OUTPUT_DIR%\

:: ============================================================================
:: Crea cartella deployment pronta all'uso
:: ============================================================================
echo.
echo [3/3] Preparazione cartella deployment: %DEPLOY_DIR%\

if exist "%DEPLOY_DIR%" rmdir /s /q "%DEPLOY_DIR%"
mkdir "%DEPLOY_DIR%"

:: Copia l'exe e le librerie bundlate
xcopy /e /i /q "%OUTPUT_DIR%\*" "%DEPLOY_DIR%\" >nul
echo   - Copiato: exe + _internal\ (framework bundlato)

:: Copia etc/config (configurazioni YAML)
if exist "etc\config" (
    xcopy /e /i /q "etc\config" "%DEPLOY_DIR%\etc\config\" >nul
    echo   - Copiato: etc\config\
    echo     ATTENZIONE: aggiornare i path assoluti in aliases.yaml e nei config YAML!
) else (
    echo   - WARN: etc\config\ non trovata, skip.
)

:: user_strategy NON viene copiata automaticamente:
:: l'utente copia SOLO le strategie che gli servono.
echo   - user_strategy\: NON copiata automaticamente.
echo     Copia a mano solo le strategie necessarie in %DEPLOY_DIR%\user_strategy\

echo.
echo ================================================================================
echo Build completata con successo!
echo ================================================================================
echo.
echo STRUTTURA DEPLOYMENT (%DEPLOY_DIR%\):
echo   run-strategy.exe
echo   _internal\               ^<-- librerie Python (non toccare)
echo   user_strategy\           ^<-- COPIA A MANO le strategie che ti servono
echo   etc\config\              ^<-- aggiornare i path nei file YAML
echo.
echo PRIMO AVVIO:
echo   cd %DEPLOY_DIR%
echo   run-strategy.exe --list
echo.
echo AVVIO CON STRATEGIA:
echo   run-strategy.exe ^<nome_config^>
echo   run-strategy.exe C:\path\assoluto\config.yaml
echo.
echo VARIABILE D'AMBIENTE (alternativa al path):
echo   set MARKET_MONITOR_CONFIG=C:\path\config.yaml
echo   run-strategy.exe
echo ================================================================================

endlocal
