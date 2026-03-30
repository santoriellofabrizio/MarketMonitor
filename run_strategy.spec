# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file per run-strategy - APPROCCIO IBRIDO

Questo file crea un eseguibile standalone per il runner delle strategie.

============================================================================
APPROCCIO IBRIDO: EXE (framework core) + Strategie Esterne (file .py)
============================================================================

1. BUILD (su computer di sviluppo):
   pyinstaller run_strategy.spec

2. DEPLOYMENT (su computer target):
   - Copia dist/run-strategy/ (eseguibile + librerie)
   - Copia user_strategy/ (le strategie che ti servono, non tutte)
   - Copia etc/config/ (le configurazioni corrispondenti)

3. STRUTTURA FINALE:
   C:\\Deployment\\
   ├── run-strategy.exe          # Eseguibile bundled
   ├── _internal\\               # Librerie Python (generate da PyInstaller)
   ├── user_strategy\\           # Strategie ESTERNE (non nel bundle)
   │   ├── equity\\
   │   ├── fixed_income\\
   │   └── ...
   └── etc\\config\\
       ├── aliases.yaml          # AGGIORNARE i path dopo il deployment!
       └── *.yaml

4. CONFIGURAZIONE:
   Nel file YAML, usa path ASSOLUTI per package_path:

   load_strategy_info:
     package_path: C:\\Deployment\\user_strategy
     module_name: \\equity\\NavChecking\\NavChecking
     class_name: NavChecking

   Oppure usa la variabile d'ambiente:
   set MARKET_MONITOR_CONFIG=C:\\Deployment\\etc\\config\\myconfig.yaml

============================================================================

PERCHE' ONEDIR E NON ONEFILE:
- ONEDIR: startup immediato, cartella _internal/ con le librerie.
  Ideale per applicazioni a terminale avviate da script/schedulatori.
- ONEFILE: singolo .exe ma avvio lento (estrae in %TEMP% ad ogni run)
  e problemi con path relativi.

NOTE DIPENDENZE SPECIALI:
- numpy: collect_all necessario per le estensioni C (.pyd/.so)
- pika: collect_all per i plugin di connessione (blocco ImportError)
- jaraco: namespace package con sottomoduli multipli, collect_all obbligatorio
- market_monitor: collect_submodules per evitare ModuleNotFoundError su
  import lazy/condizionali non tracciabili dallo static analysis
- tkinter: incluso perche' alcune strategie usano GUI Tkinter
"""

import os
import sys
from pathlib import Path

# Path del progetto (directory dove si trova questo .spec)
project_root = Path(SPECPATH)

# ============================================================================
# 0. collect_all / collect_submodules per pacchetti problematici
# ============================================================================
from PyInstaller.utils.hooks import collect_all, collect_submodules

# numpy: le estensioni C non vengono trovate automaticamente
datas_numpy, binaries_numpy, hiddenimports_numpy = collect_all('numpy')

# pika: i plugin (connection parameters, adapters, ecc.) sono lazy-loaded
datas_pika, binaries_pika, hiddenimports_pika = collect_all('pika')

# jaraco: namespace package frammentato in piu' distribuzioni
datas_jaraco, binaries_jaraco, hiddenimports_jaraco = collect_all('jaraco')

# market_monitor: collect_submodules garantisce che TUTTI i sottomoduli
# siano inclusi nell'archivio, indipendentemente da come vengono importati
# a runtime (import condizionali, lazy, importlib, ecc.).
hiddenimports_mm = collect_submodules('market_monitor')

# sfm_datalibrary / sfm_data_provider: librerie interne Sella impacchettate,
# non rilevabili dallo static analysis di PyInstaller.
datas_sfm_dl, binaries_sfm_dl, hiddenimports_sfm_dl = collect_all('sfm_datalibrary')
# sfm_data_provider e' opzionale in questo bundle (usata solo da user_strategy/).
# Se non e' ancora installata nel virtualenv di build, la skip senza errori.
try:
    datas_sfm_dp, binaries_sfm_dp, hiddenimports_sfm_dp = collect_all('sfm_data_provider')
except Exception:
    datas_sfm_dp, binaries_sfm_dp, hiddenimports_sfm_dp = [], [], []

# ============================================================================
# 1. Hidden imports
# ============================================================================
hiddenimports = [
    # ---- market_monitor: tutti i sottomoduli raccolti automaticamente ----
    *hiddenimports_mm,

    # ---- PyQt5 (usato da alcune strategie e dalla control panel) ----
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'PyQt5.QtChart',
    'PyQt5.sip',

    # ---- Tkinter (usato da alcune strategie per GUI tabellare) ----
    'tkinter',
    'tkinter.ttk',
    '_tkinter',

    # ---- Matplotlib ----
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',

    # ---- Librerie dati ----
    'pandas',
    'scipy',
    'pyarrow',
    'openpyxl',
    'fastavro',

    # ---- Infrastruttura ----
    'redis',
    'aiosqlite',
    'ruamel.yaml',
    'ruamel.yaml.clib',
    'watchdog',
    'watchdog.observers',
    'watchdog.observers.polling',
    'tqdm',
    'joblib',
    'questionary',
    'httpx',
    'authlib',

    # ---- Bloomberg (opzionale, caricato condizionalmente) ----
    'blpapi',
    'xbbg',

    # ---- Kafka ----
    'confluent_kafka',

    # ---- Sella internals ----
    'sfm_dbconnections',
    'sfm_dbconnections.DbConnectionParameters',
    *hiddenimports_sfm_dl,
    *hiddenimports_sfm_dp,

    # ---- Da collect_all ----
    *hiddenimports_numpy,
    *hiddenimports_pika,
    *hiddenimports_jaraco,
]

# ============================================================================
# 2. Data files
# ============================================================================
# NOTA: user_strategy/ ed etc/config/ NON vengono inclusi nel bundle.
# Devono essere distribuiti separatamente e posizionati accanto all'exe.
datas = [
    # Da collect_all (metadati e risorse dei pacchetti)
    *datas_numpy,
    *datas_pika,
    *datas_jaraco,
    *datas_sfm_dl,
    *datas_sfm_dp,
]

# ============================================================================
# 3. Binaries e librerie native
# ============================================================================
binaries = [
    *binaries_numpy,
    *binaries_pika,
    *binaries_jaraco,
    *binaries_sfm_dl,
    *binaries_sfm_dp,
]

# ============================================================================
# 4. Analysis - Analisi dell'entrypoint
# ============================================================================
a = Analysis(
    [str(project_root / 'src' / 'market_monitor' / 'entry' / 'run_strategy.py')],
    pathex=[
        str(project_root / 'src'),
        str(project_root),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'notebook',
        'jupyter',
        'sphinx',
        'pytest',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ============================================================================
# 5. PYZ - Archivio Python compresso
# ============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ============================================================================
# 6. EXE + COLLECT - Modalita' ONEDIR (cartella con _internal/)
# ============================================================================
# ONEDIR: exclude_binaries=True nell'EXE, poi COLLECT raduna tutto.
# Risultato: dist/run-strategy/run-strategy.exe + dist/run-strategy/_internal/
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,   # <-- True = onedir (binaries vanno in COLLECT)
    name='run-strategy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,            # Applicazione a terminale: mostra output/log
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run-strategy',     # <-- nome della cartella in dist/
)

print("\n" + "=" * 80)
print("Strategy Runner Build: MarketMonitor (ONEDIR - Approccio Ibrido)")
print("=" * 80)
print(f"Output: dist/run-strategy/run-strategy.exe")
print("\nSTRUTTURA DEPLOYMENT:")
print("  C:\\Deployment\\")
print("  +-- run-strategy.exe")
print("  +-- _internal\\         (librerie bundlate)")
print("  +-- user_strategy\\     (copiare solo le strategie necessarie)")
print("  +-- etc\\config\\        (copiare le config YAML)")
print("\nUSO:")
print("  run-strategy.exe <nome_config>")
print("  run-strategy.exe --list")
print("  run-strategy.exe C:\\path\\assoluto\\config.yaml")
print("=" * 80)
