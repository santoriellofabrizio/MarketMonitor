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
"""

import os
import sys
from pathlib import Path

# Path del progetto
project_root = Path(SPECPATH)

# ============================================================================
# 0. collect_all per pacchetti problematici
# ============================================================================
from PyInstaller.utils.hooks import collect_all, collect_submodules

# numpy: le estensioni C non vengono trovate automaticamente
datas_numpy, binaries_numpy, hiddenimports_numpy = collect_all('numpy')

# pika: i plugin (connection parameters, adapters, ecc.) sono lazy-loaded
datas_pika, binaries_pika, hiddenimports_pika = collect_all('pika')

# jaraco: namespace package frammentato in più distribuzioni
datas_jaraco, binaries_jaraco, hiddenimports_jaraco = collect_all('jaraco')

# ============================================================================
# 1. Hidden imports - SOLO market_monitor core + dashboard
# ============================================================================
hiddenimports = [
    # ---- Entry / Runner ----
    'market_monitor.entry._base',
    'market_monitor.entry.run_dashboard',

    # ---- Core ----
    'market_monitor.builder',
    'market_monitor.live_data_hub.real_time_data_hub',
    'market_monitor.live_data_hub.data_store',
    'market_monitor.live_data_hub.live_subscription',
    'market_monitor.live_data_hub.subscription_service',

    # ---- Input threads ----
    'market_monitor.input_threads.bloomberg',
    'market_monitor.input_threads.redis',
    'market_monitor.input_threads.trade',
    'market_monitor.input_threads.excel',
    'market_monitor.input_threads.kafka',
    'market_monitor.input_threads.event_handler.BBGEventHandler',

    # ---- Publishers ----
    'market_monitor.publishers.base',
    'market_monitor.publishers.redis_publisher',
    'market_monitor.publishers.timeseries_publisher',

    # ---- Strategy ----
    'market_monitor.strategy.strategy_ui.StrategyUI',
    'market_monitor.strategy.strategy_ui.StrategyUIAsync',
    'market_monitor.strategy.common.trade_manager.trade_manager',
    'market_monitor.strategy.common.trade_manager.book_memory',
    'market_monitor.strategy.common.trade_manager.time_zero_pl',
    'market_monitor.strategy.common.trade_manager.flow_detector',
    'market_monitor.strategy.common.trade_manager.trade_templates',

    # ---- GUI - PyQt5 Dashboard (tutti i moduli) ----
    'market_monitor.gui.implementations.PyQt5Dashboard.builder',
    'market_monitor.gui.implementations.PyQt5Dashboard.trade_dashboard',
    'market_monitor.gui.implementations.PyQt5Dashboard.base',
    'market_monitor.gui.implementations.PyQt5Dashboard.common',
    'market_monitor.gui.implementations.PyQt5Dashboard.dashboard_extension',
    'market_monitor.gui.implementations.PyQt5Dashboard.detached_windows',
    'market_monitor.gui.implementations.PyQt5Dashboard.metrics_definition',
    'market_monitor.gui.implementations.PyQt5Dashboard.worker_thread',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.chart_widget',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.dashboard_state',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.flow_monitor_widget',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.trade_history_window',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.pivot_table',
    'market_monitor.gui.implementations.PyQt5Dashboard.widgets.trade_table',

    # ---- GUI - Legacy (potrebbe essere referenziato da builder) ----
    'market_monitor.gui.implementations.GUI',
    'market_monitor.gui.threaded_GUI.GUIQueue',
    'market_monitor.gui.threaded_GUI.QueueDataSource',

    # ---- Utils ----
    'market_monitor.utils.config_helpers',
    'market_monitor.utils.config_observer',
    'market_monitor.utils.decorators',

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
    'sfm_dbconnections',
    'sfm_dbconnections.DbConnectionParameters',

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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Aggiungi un .ico se vuoi
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
print("  oppure usa la variabile d'ambiente:")
print("  set MARKET_MONITOR_CONFIG=C:\\MieiConfig\\dashboard.yaml")
print("=" * 80)
