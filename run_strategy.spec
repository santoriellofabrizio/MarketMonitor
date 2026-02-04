# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file per run-strategy - APPROCCIO IBRIDO

Questo file crea un eseguibile standalone con SOLO il core di MarketMonitor.
Le strategie user_strategy sono caricate come file .py esterni.

============================================================================
APPROCCIO IBRIDO: EXE + Strategie Esterne
============================================================================

1. BUILD (su computer di sviluppo):
   pyinstaller run_strategy.spec

2. DEPLOYMENT (su computer target):
   - Copia dist/run-strategy/ (eseguibile + librerie)
   - Copia user_strategy/ (strategie come .py)
   - Copia etc/config/ (configurazioni)

3. STRUTTURA FINALE:
   C:\Deployment\
   ├── run-strategy.exe           # Eseguibile bundled
   ├── _internal/                 # Librerie Python
   ├── user_strategy/             # Strategie .py (ESTERNE)
   │   ├── equity/
   │   ├── fixed_income/
   │   └── utils/
   └── etc/config/
       └── my_config.yaml

4. CONFIGURAZIONE:
   Nel file YAML, usa path relativi:

   load_strategy_info:
     package_path: ./user_strategy/equity/LiveAnalysis
     module_name: EquityLiveAnalysisStrategy
     class_name: EquityLiveAnalysisStrategy

============================================================================

Vantaggi:
- ✅ EXE standalone (no Python richiesto)
- ✅ Strategie modificabili senza ricompilare
- ✅ File più piccolo (~100-200MB invece di 500MB)
- ✅ Facile aggiornamento strategie (copia nuovi .py)
"""

import os
import sys
from pathlib import Path

# Path del progetto
project_root = Path(SPECPATH)

# ============================================================================
# 1. Hidden imports - SOLO market_monitor core
# ============================================================================
hiddenimports = [
    # Market monitor core (NO user_strategy!)
    'market_monitor.strategy.StrategyUI.StrategyUI',
    'market_monitor.strategy.StrategyUI.StrategyUIAsync',
    'market_monitor.publishers.redis_publisher',
    'market_monitor.publishers.timeseries_publisher',
    'market_monitor.live_data_hub.real_time_data_hub',
    'market_monitor.input_threads.bloomberg',
    'market_monitor.input_threads.redis',
    'market_monitor.input_threads.trade',
    'market_monitor.input_threads.excel',
    'market_monitor.gui.implementations.GUI',
    'market_monitor.gui.threaded_GUI.GUIQueue',
    'market_monitor.gui.threaded_GUI.QueueDataSource',
    'market_monitor.gui.threaded_GUI.ThreadGUIExcel',
    'market_monitor.gui.threaded_GUI.TradeThreadGUITkinter',
    'market_monitor.input_threads.event_handler.BBGEventHandler',
    'market_monitor.builder',

    # Librerie esterne essenziali
    'pandas',
    'numpy',
    'redis',
    'ruamel.yaml',
    'scipy',
    'matplotlib',
    'openpyxl',
    'pyarrow',
    'questionary',
    'watchdog',
    'tqdm',
    'joblib',
    'aiosqlite',

    # PyQt5 (se usato)
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',

    # Bloomberg (se disponibile - opzionale)
    'blpapi',
    'xbbg',

    # Dipendenze interne Sella
    'sfm_dbconnections',
    'sfm_dbconnections.DbConnectionParameters',
]

# ============================================================================
# 2. Data files - SOLO configurazioni (NO user_strategy!)
# ============================================================================
datas = [
    # Configurazioni YAML template
    (str(project_root / 'etc' / 'config'), 'etc/config'),

    # NOTE: user_strategy NON è inclusa!
    # Le strategie saranno caricate come file .py esterni
]

# ============================================================================
# 3. Binaries e librerie native
# ============================================================================
binaries = []

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
    hookspath=[],  # No custom hooks needed
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Escludi moduli inutili per ridurre dimensione
        'tkinter',
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
# 6. EXE - Eseguibile finale
# ============================================================================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run-strategy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Aggiungi un .ico se vuoi
)

# ============================================================================
# 7. COLLECT - Raccolta di tutti i file in una directory
# ============================================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run-strategy',
)

print("\n" + "=" * 80)
print("✅ HYBRID BUILD: MarketMonitor Core + External Strategies")
print("=" * 80)
print(f"Output: dist/run-strategy/")
print(f"Eseguibile: dist/run-strategy/run-strategy{'.exe' if sys.platform == 'win32' else ''}")
print("\n📦 DEPLOYMENT STEPS:")
print("  1. Copia dist/run-strategy/ sul computer target")
print("  2. Copia user_strategy/ (strategie .py) accanto all'exe")
print("  3. Copia etc/config/ con le configurazioni")
print("  4. Usa path relativi nei config YAML (es: ./user_strategy/equity/LiveAnalysis)")
print("=" * 80)
