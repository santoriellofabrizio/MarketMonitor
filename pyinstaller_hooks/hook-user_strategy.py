"""
PyInstaller hook per user_strategy

Questo hook assicura che tutti i moduli in user_strategy siano inclusi
nel bundle, anche se caricati dinamicamente.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Raccolta automatica di tutti i submoduli
datas, binaries, hiddenimports = collect_all('user_strategy')

# Aggiungi esplicitamente tutti i submoduli
hiddenimports += collect_submodules('user_strategy')

print(f"[hook-user_strategy] Trovati {len(hiddenimports)} hidden imports")
