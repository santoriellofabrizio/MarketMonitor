# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file per run-dashboard - APPROCCIO IBRIDO

Questo file crea un eseguibile standalone per la dashboard PyQt5.

============================================================================
APPROCCIO IBRIDO: EXE + Strategie Esterne
============================================================================

1. BUILD (su computer di sviluppo):
   pyinstaller run_dashboard.spec

2. DEPLOYMENT (su computer target):
   - Copia dist/run-dashboard/ (eseguibile + librerie)
   - Copia etc/config/ (configurazioni)

3. STRUTTURA FINALE:
   C:\Deployment\
   ├── run-dashboard.exe          # Eseguibile bundled
   ├── _internal/                 # Librerie Python
   └── etc/config/
       └── dashboard.yaml

4. CONFIGURAZIONE:
   Nel file YAML, usa path relativi:

   dashboard:
     ...

============================================================================

NOTE DIPENDENZE SPECIALI:
- numpy: collect_all necessario per le estensioni C (.pyd/.so)
- pika: collect_all per i plugin di connessione (blocco ImportError)
- jaraco: namespace package con sottomoduli multipli, collect_all obbligatorio
- market_monitor: collect_submodules per evitare ModuleNotFoundError su
  import lazy/condizionali non tracciabili dallo static analysis
"""

import os
import sys
from pathlib import Path

# Path del progetto
project_root = Path(SPECPATH)

# ============================================================================
# 0. collect_all / collect_submodules per pacchetti problematici
# ============================================================================
from PyInstaller.utils.hooks import collect_all, collect_submodules

# numpy: le estensioni C non vengono trovate automaticamente
datas_numpy, binaries_numpy, hiddenimports_numpy = collect_all('numpy')

# pika: i plugin (connection parameters, adapters, ecc.) sono lazy-loaded
datas_pika, binaries_pika, hiddenimports_pika = collect_all('pika')

# jaraco: namespace package frammentato in più distribuzioni
datas_jaraco, binaries_jaraco, hiddenimports_jaraco = collect_all('jaraco')

# market_monitor: collect_submodules garantisce che TUTTI i sottomoduli
# siano inclusi nell'archivio, indipendentemente da come vengono importati
# a runtime (import condizionali, lazy, importlib, ecc.).
# Questo risolve ModuleNotFoundError su worker_thread e altri moduli
# non raggiungibili dallo static analysis di PyInstaller.
hiddenimports_mm = collect_submodules('market_monitor')

# SFM internal libraries: collect_submodules per includere tutti i sottomoduli
# di ciascun pacchetto interno Sella Financial Markets.
hiddenimports_sfm_datalibrary = collect_submodules('sfm_datalibrary')
hiddenimports_sfm_dbconnections = collect_submodules('sfm_dbconnections')
hiddenimports_sfm_data_provider = collect_submodules('sfm_data_provider')
hiddenimports_sfm_quantlib = collect_submodules('sfm_quantlib')
hiddenimports_sfm_return_adjustments_lib = collect_submodules('sfm_return_adjustments_lib')
hiddenimports_sfm_pcf_db_library = collect_submodules('sfm_pcf_db_library')
hiddenimports_sfm_timescaledb_queries = collect_submodules('sfm_timescaledb_queries')

# ============================================================================
# 1. Hidden imports
# ============================================================================
hiddenimports = [
    # ---- market_monitor: tutti i sottomoduli raccolti automaticamente ----
    *hiddenimports_mm,

    # ---- PyQt5 ----
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'PyQt5.QtChart',
    'PyQt5.sip',

    # ---- Matplotlib (usato nei widget grafici) ----
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_agg',
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

    # ---- Bloomberg (opzionale) ----
    'blpapi',
    'xbbg',

    # ---- Kafka ----
    'confluent_kafka',

    # ---- Sella internals ----
    *hiddenimports_sfm_datalibrary,
    *hiddenimports_sfm_dbconnections,
    *hiddenimports_sfm_data_provider,
    *hiddenimports_sfm_quantlib,
    *hiddenimports_sfm_return_adjustments_lib,
    *hiddenimports_sfm_pcf_db_library,
    *hiddenimports_sfm_timescaledb_queries,

    # ---- Da collect_all ----
    *hiddenimports_numpy,
    *hiddenimports_pika,
    *hiddenimports_jaraco,
]

# ============================================================================
# 2. Data files
# ============================================================================
datas = [
    # Configurazioni YAML template
    (str(project_root / 'etc' / 'config'), 'etc/config'),

    # Dati di esempio
    (str(project_root / 'etc' / 'data'), 'etc/data'),

    # Da collect_all
    *datas_numpy,
    *datas_pika,
    *datas_jaraco,
]

# ============================================================================
# 3. Binaries e librerie native
# ============================================================================
binaries = [
    *binaries_numpy,
    *binaries_pika,
    *binaries_jaraco,
]

# ============================================================================
# 4. Analysis - Analisi dell'entrypoint
# ============================================================================
a = Analysis(
    [str(project_root / 'src' / 'market_monitor' / 'entry' / 'run_dashboard.py')],
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
        # Riduci dimensione rimuovendo ciò che non serve alla dashboard
        'IPython',
        'notebook',
        'jupyter',
        'sphinx',
        'pytest',
        'setuptools',
        # tkinter non serve (dashboard è PyQt5)
        'tkinter',
        '_tkinter',
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
# 6. EXE - Eseguibile ONEFILE (tutto in un singolo .exe)
# ============================================================================
# ONEFILE: exclude_binaries=False + a.binaries/zipfiles/datas dentro EXE
# All'avvio si estrae in %TEMP%\_MEIxxxxx (startup ~5-10s più lento)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,    # <-- inclusi nel singolo exe
    a.zipfiles,    # <-- inclusi nel singolo exe
    a.datas,       # <-- inclusi nel singolo exe
    exclude_binaries=False,  # <-- False = onefile
    name='run-dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # console=False per nascondere il terminale (finestra GUI pura)
    # console=True  per vedere i log a terminale (utile in sviluppo)
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r'C:\AFMachineLearning\Libraries\MarketMonitor\build\tradingDashboardIcon.ico',
)

# COLLECT non serve in modalità onefile
# (decommentare e commentare EXE sopra per tornare alla modalità cartella)

print("\n" + "=" * 80)
print("Dashboard Build: MarketMonitor PyQt5 Dashboard (ONEFILE)")
print("=" * 80)
print(f"Output: dist/run-dashboard.exe")
print("\nUSO:")
print("  ATTENZIONE: in onefile il config viene cercato relativo al temp dir.")
print("  Passa SEMPRE il path assoluto al config:")
print(r'  run-dashboard.exe "C:\MieiConfig\dashboard.yaml"')
print("  oppure usa la variabile d\'ambiente:")
print("  set MARKET_MONITOR_CONFIG=C:\\MieiConfig\\dashboard.yaml")
print("=" * 80)
