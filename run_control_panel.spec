# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file per run-control-panel - ONEFILE

Questo file crea un eseguibile standalone per il Control Panel di MarketMonitor.

============================================================================
ARCHITETTURA
============================================================================

run-control-panel.exe e' un ONEFILE:
- Si avvia una sola volta dall'utente
- Startup piu' lento accettabile (nessun conflitto di _internal\ con run-strategy)
- Lancia run-strategy.exe come sottoprocesso (QProcess) dalla stessa cartella
- Comunica con la strategia esclusivamente via Redis

STRUTTURA DEPLOYMENT:
   C:\Deployment\
   +-- run-strategy.exe         (ONEDIR, avviato da run-control-panel via QProcess)
   +-- run-control-panel.exe    (ONEFILE, avviato dall'utente)
   +-- run-dashboard.exe        (ONEFILE, viewer dati standalone)
   +-- _internal\               (librerie di run-strategy)
   +-- user_strategy\           (strategie .py, copiate a mano)
   +-- etc\config\              (file YAML, copiati a mano)

USO:
   run-control-panel.exe [--config <nome_config>] [--host <redis_host>]

NOTE DIPENDENZE SPECIALI:
- sfm_datalibrary / sfm_data_provider: collect_all per includere tutti i moduli
- numpy / pika / jaraco: collect_all per estensioni C e namespace package
- market_monitor: collect_submodules per import dinamici non tracciabili
- tkinter escluso: il panel usa PyQt5, non Tkinter
"""

import os
import sys
from pathlib import Path

project_root = Path(SPECPATH)

# ============================================================================
# 0. collect_all / collect_submodules per pacchetti problematici
# ============================================================================
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas_numpy, binaries_numpy, hiddenimports_numpy = collect_all('numpy')
datas_pika, binaries_pika, hiddenimports_pika = collect_all('pika')
datas_jaraco, binaries_jaraco, hiddenimports_jaraco = collect_all('jaraco')
hiddenimports_mm = collect_submodules('market_monitor')
datas_sfm_dl, binaries_sfm_dl, hiddenimports_sfm_dl = collect_all('sfm_datalibrary')
datas_sfm_dp, binaries_sfm_dp, hiddenimports_sfm_dp = collect_all('sfm_data_provider')

# ============================================================================
# 1. Hidden imports
# ============================================================================
hiddenimports = [
    *hiddenimports_mm,

    # ---- PyQt5 ----
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',
    'PyQt5.QtChart',
    'PyQt5.sip',

    # ---- Matplotlib (widget grafici nel panel) ----
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
datas = [
    *datas_numpy,
    *datas_pika,
    *datas_jaraco,
    *datas_sfm_dl,
    *datas_sfm_dp,
]

# ============================================================================
# 3. Binaries
# ============================================================================
binaries = [
    *binaries_numpy,
    *binaries_pika,
    *binaries_jaraco,
    *binaries_sfm_dl,
    *binaries_sfm_dp,
]

# ============================================================================
# 4. Analysis
# ============================================================================
a = Analysis(
    [str(project_root / 'src' / 'market_monitor' / 'entry' / 'run_control_panel.py')],
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
        'tkinter',
        '_tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ============================================================================
# 5. PYZ
# ============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ============================================================================
# 6. EXE ONEFILE
# ============================================================================
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    exclude_binaries=False,
    name='run-control-panel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # GUI pura, nessuna finestra terminale
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

print("\n" + "=" * 80)
print("Control Panel Build: MarketMonitor (ONEFILE)")
print("=" * 80)
print(f"Output: dist/run-control-panel.exe")
print("\nUSO:")
print("  run-control-panel.exe")
print("  run-control-panel.exe --config <nome_config>")
print("  run-control-panel.exe --host <redis_host> --port <redis_port>")
print("\nNOTA: run-strategy.exe deve essere nella stessa cartella.")
print("=" * 80)
