"""
CLI package per MarketMonitor.

Struttura:
- _base.py: Logica comune CLI (parsing, setup, shutdown)
- run_strategy.py: Entry point per strategia
- run_mock.py: Entry point per mock
- run_dashboard.py: Entry point per dashboard
"""
import os
import sys
from pathlib import Path

# Root del progetto (calcolata dal path del package, non dalla CWD)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Aggiungi la root del progetto al path per trovare user_strategy
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Cambia la working directory alla root del progetto
# Questo garantisce che i path relativi nelle config funzionino
os.chdir(_PROJECT_ROOT)


def get_project_root() -> Path:
    """Ritorna la root del progetto."""
    return _PROJECT_ROOT
