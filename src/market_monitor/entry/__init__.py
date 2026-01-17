"""
Entry package per MarketMonitor.

Struttura:
- _base.py: Logica comune CLI (parsing, setup, shutdown)
- run_strategy.py: Entry point per strategia
- run_mock.py: Entry point per mock
- run_dashboard.py: Entry point per dashboard

Variabili d'ambiente supportate:
- MARKET_MONITOR_STRATEGIES: Path aggiuntivi per le strategie (separati da ;)
- MARKET_MONITOR_CONFIG: Config di default
"""
import os
import sys
from pathlib import Path

# Root del progetto (calcolata dal path del package, non dalla CWD)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Aggiungi la root del progetto al path per trovare user_strategy
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Aggiungi user_strategy esplicitamente
_USER_STRATEGY = _PROJECT_ROOT / "user_strategy"
if _USER_STRATEGY.exists() and str(_USER_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_USER_STRATEGY))

# Supporto per path strategie aggiuntivi via env var
_EXTRA_STRATEGIES = os.environ.get("MARKET_MONITOR_STRATEGIES", "")
if _EXTRA_STRATEGIES:
    for path in _EXTRA_STRATEGIES.split(";"):
        path = path.strip()
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)

# Cambia la working directory alla root del progetto
os.chdir(_PROJECT_ROOT)


def get_project_root() -> Path:
    """Ritorna la root del progetto."""
    return _PROJECT_ROOT


def get_strategy_paths() -> list[Path]:
    """Ritorna tutti i path dove vengono cercate le strategie."""
    paths = [_PROJECT_ROOT, _USER_STRATEGY]
    if _EXTRA_STRATEGIES:
        paths.extend(Path(p.strip()) for p in _EXTRA_STRATEGIES.split(";") if p.strip())
    return [p for p in paths if p.exists()]
