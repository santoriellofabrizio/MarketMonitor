# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file per run-strategy

Uso:
    pyinstaller run_strategy.spec

Questo file gestisce correttamente:
- Import dinamici di user_strategy
- Configurazioni YAML
- Dipendenze native (Bloomberg, PyQt5, etc.)
"""

import os
import sys
from pathlib import Path

# Path del progetto
project_root = Path(SPECPATH)
user_strategy_path = project_root / "user_strategy"

# ============================================================================
# 1. Raccolta automatica di tutti i moduli user_strategy
# ============================================================================
def collect_user_strategy_modules(base_path):
    """Trova tutti i moduli Python in user_strategy per hidden imports."""
    modules = []
    for root, dirs, files in os.walk(base_path):
        # Salta __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # Converti path in module name
                rel_path = Path(root).relative_to(project_root)
                module_name = str(rel_path / file[:-3]).replace(os.sep, '.')
                modules.append(module_name)

    return modules

user_strategy_modules = collect_user_strategy_modules(user_strategy_path)

print("=" * 80)
print("MODULI USER_STRATEGY RILEVATI:")
print("=" * 80)
for mod in sorted(user_strategy_modules):
    print(f"  - {mod}")
print(f"\nTotale: {len(user_strategy_modules)} moduli")
print("=" * 80)

# ============================================================================
# 2. Hidden imports (moduli non rilevati automaticamente)
# ============================================================================
hiddenimports = [
    # User strategy modules (caricati dinamicamente)
    *user_strategy_modules,

    # Market monitor core
    'market_monitor.strategy.StrategyUI.StrategyUI',
    'market_monitor.strategy.StrategyUI.StrategyUIAsync',
    'market_monitor.publishers.redis_publisher',
    'market_monitor.publishers.timeseries_publisher',
    'market_monitor.live_data_hub.real_time_data_hub',
    'market_monitor.input_threads.bloomberg',
    'market_monitor.input_threads.redis',
    'market_monitor.input_threads.trade',
    'market_monitor.input_threads.excel',

    # Librerie esterne che potrebbero non essere rilevate
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

    # PyQt5
    'PyQt5.QtCore',
    'PyQt5.QtWidgets',
    'PyQt5.QtGui',

    # Bloomberg (se disponibile)
    'blpapi',
    'xbbg',
]

# ============================================================================
# 3. Data files (configurazioni, user_strategy come moduli)
# ============================================================================
datas = [
    # Configurazioni YAML
    (str(project_root / 'etc' / 'config'), 'etc/config'),

    # User strategy come source files (per caricamento dinamico)
    (str(user_strategy_path), 'user_strategy'),

    # README se serve
    # (str(project_root / 'README.md'), '.'),
]

# ============================================================================
# 4. Binaries e librerie native
# ============================================================================
binaries = []

# ============================================================================
# 5. Analysis - Analisi dell'entrypoint
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
    hookspath=[str(project_root / 'pyinstaller_hooks')],  # Hook personalizzati
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Escludi moduli inutili per ridurre dimensione
        'tkinter',  # se non usi Tkinter
        # 'IPython',
        # 'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ============================================================================
# 6. PYZ - Archivio Python compresso
# ============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ============================================================================
# 7. EXE - Eseguibile finale
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
# 8. COLLECT - Raccolta di tutti i file in una directory
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
print("✅ Spec file configurato correttamente!")
print("=" * 80)
print(f"Output: dist/run-strategy/")
print(f"Eseguibile: dist/run-strategy/run-strategy{'exe' if sys.platform == 'win32' else ''}")
print("=" * 80)
