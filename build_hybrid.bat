@echo off
REM ============================================================================
REM Build script per MarketMonitor - Approccio Ibrido (Windows)
REM ============================================================================
REM
REM Questo script crea un deployment package completo:
REM - Eseguibile PyInstaller (core MarketMonitor)
REM - Strategie user_strategy come .py
REM - Configurazioni
REM
REM Uso:
REM   build_hybrid.bat
REM   build_hybrid.bat --clean  (pulisce build precedenti)
REM
REM Output: deployment\ directory pronta per la distribuzione
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================================
echo MarketMonitor - Hybrid Deployment Builder
echo ============================================================================

REM Parse arguments
set CLEAN=0
if "%1"=="--clean" set CLEAN=1

REM 1. Clean (se richiesto)
if %CLEAN%==1 (
    echo.
    echo [CLEAN] Pulizia build precedenti...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist deployment rmdir /s /q deployment
    for /r %%i in (*.pyc) do del "%%i"
    for /d /r %%i in (__pycache__) do if exist "%%i" rmdir /s /q "%%i"
)

REM 2. Check dependencies
echo.
echo [CHECK] Verifica dipendenze...
where pyinstaller >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller non trovato!
    echo [INFO] Installazione in corso...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [ERROR] Installazione PyInstaller fallita!
        exit /b 1
    )
)

REM 3. Build with PyInstaller
echo.
echo [BUILD] Build eseguibile con PyInstaller...
pyinstaller run_strategy.spec
if %errorlevel% neq 0 (
    echo [ERROR] Build fallito!
    exit /b 1
)

if not exist "dist\run-strategy\run-strategy.exe" (
    echo [ERROR] Eseguibile non trovato!
    exit /b 1
)

echo [OK] Build completato!

REM 4. Create deployment package
echo.
echo [DEPLOY] Creazione deployment package...
if exist deployment rmdir /s /q deployment
mkdir deployment

REM Copy executable
echo   - Copia eseguibile...
xcopy /E /I /Q dist\run-strategy deployment\run-strategy

REM Copy user_strategy (as .py files)
echo   - Copia strategie (user_strategy)...
xcopy /E /I /Q user_strategy deployment\run-strategy\user_strategy

REM Copy config examples
echo   - Copia configurazioni...
mkdir deployment\run-strategy\etc\config
copy etc\config\config_hybrid_deployment_example.yaml deployment\run-strategy\etc\config\
copy etc\config\config_template.yaml deployment\run-strategy\etc\config\

REM Create README for deployment
echo   - Crea README...
(
echo ============================================================================
echo MarketMonitor - Hybrid Deployment Package
echo ============================================================================
echo.
echo CONTENUTO:
echo   - run-strategy.exe      Eseguibile standalone
echo   - _internal\            Librerie Python bundled
echo   - user_strategy\        Strategie Python ^(modificabili^)
echo   - etc\config\           Configurazioni YAML
echo.
echo USO:
echo   1. Modifica etc\config\config_hybrid_deployment_example.yaml
echo      - Cambia load_strategy_info per scegliere la strategia
echo      - Configura data sources ^(Bloomberg, Redis, etc.^)
echo.
echo   2. Esegui:
echo      run-strategy.exe etc\config\config_hybrid_deployment_example.yaml
echo.
echo      Oppure rinomina il config e usa l'alias:
echo      run-strategy.exe my_config
echo.
echo STRATEGIE DISPONIBILI:
echo   - user_strategy\test_strategy\SimplePriceMonitorStrategy.py
echo   - user_strategy\test_strategy\TestTradeManagerStrategy.py
echo   - user_strategy\equity\LiveAnalysis\
echo   - user_strategy\fixed_income\
echo.
echo PERSONALIZZAZIONE:
echo   Le strategie in user_strategy\ sono file .py modificabili.
echo   Puoi editarli senza dover ricompilare l'eseguibile!
echo.
echo REQUISITI:
echo   - Nessuno! L'eseguibile include Python e tutte le dipendenze.
echo   - Bloomberg blpapi richiede installazione separata se usi Bloomberg.
echo   - Redis server se usi redis_data_distributor.
echo.
echo LOGS:
echo   I log vengono salvati in logs\
echo.
echo ============================================================================
) > deployment\run-strategy\README_DEPLOYMENT.txt

REM 5. Create archive (optional - requires 7zip or tar)
echo.
echo [ARCHIVE] Creazione archivio...
where 7z >nul 2>nul
if %errorlevel% equ 0 (
    cd deployment
    7z a -tzip MarketMonitor-hybrid-%date:~-4%%date:~3,2%%date:~0,2%.zip run-strategy\
    cd ..
    echo [OK] Archivio creato con 7zip
) else (
    where tar >nul 2>nul
    if %errorlevel% equ 0 (
        cd deployment
        tar -czf MarketMonitor-hybrid-%date:~-4%%date:~3,2%%date:~0,2%.tar.gz run-strategy\
        cd ..
        echo [OK] Archivio creato con tar
    ) else (
        echo [WARNING] 7zip o tar non trovato - archivio non creato
        echo [INFO] Copia manualmente deployment\run-strategy\ per distribuire
    )
)

REM 6. Summary
echo.
echo ============================================================================
echo [SUCCESS] BUILD COMPLETATO!
echo ============================================================================
echo.
echo Output:
echo   - deployment\run-strategy\                  ^(directory deployment^)
echo   - deployment\MarketMonitor-hybrid-*.zip     ^(archivio distribuibile^)
echo.
echo Prossimi passi:
echo   1. Copia deployment\run-strategy\ sul computer target
echo   2. Modifica etc\config\config_hybrid_deployment_example.yaml
echo   3. Esegui: run-strategy.exe etc\config\...
echo.
echo ============================================================================

endlocal
