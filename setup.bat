@echo off
SETLOCAL EnableExtensions

echo ==========================================
echo    UV PROJECT SETUP (GBS PRO STYLE)
echo ==========================================

:: 1. Controllo/Inizializzazione TOML
if not exist "pyproject.toml" (
    echo [+] pyproject.toml non trovato. Inizializzazione in corso...
    uv init
) else (
    echo [!] pyproject.toml gia' presente. Salto uv init.
)

:: 2. Creazione Virtual Environment
if not exist ".venv" (
    echo [+] Creazione del virtual environment (.venv)...
    uv venv
) else (
    echo [!] Cartella .venv gia' esistente.
)

:: 3. Sincronizzazione con il NAS
echo [+] Sincronizzazione pacchetti dal NAS...
echo Percorso: \\nas.sg.gbs.pro\AreaFinanza\Share\PROGETTI ML\Python\python3_11\python packages

uv sync --no-index --find-links=file:"\\nas.sg.gbs.pro\AreaFinanza\Share\PROGETTI ML\Python\python3_11\python packages" -U

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo    SETUP COMPLETATO CON SUCCESSO!
    echo ==========================================
    echo Per iniziare a lavorare:
    echo 1. Apri il progetto su PyCharm
    echo 2. Seleziona l'interprete dentro .venv\Scripts\python.exe
    echo 3. Nel terminale usa: .venv\Scripts\activate
) else (
    echo.
    echo [ERRORE] Qualcosa e' andato storto durante la sincronizzazione.
    echo Verifica la connessione al NAS o rivolgiti a Giulio Merlo.
)

pause