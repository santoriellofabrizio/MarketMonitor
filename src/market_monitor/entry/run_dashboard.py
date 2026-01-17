"""
Entry point per la dashboard PyQt5.

Uso CLI:
    run-dashboard <config_name>
    run-dashboard --list
    run-dashboard --describe <config_name>
    run-dashboard --dry-run <config_name>

Uso programmatico:
    from market_monitor.entry.run_dashboard import run_dashboard
    
    run_dashboard("config_dashboard")
    run_dashboard("/path/to/config.yaml")
    run_dashboard({"name": "test", ...})
"""
import sys
from typing import List, Optional, Union
from pathlib import Path

from market_monitor.entry._base import BaseRunner


class DashboardRunner(BaseRunner):
    """Runner per la dashboard PyQt5."""
    
    name = "dashboard"
    description = "Dashboard PyQt5 per MarketMonitor"
    log_file = "dashboard_debug.log"
    
    def __init__(self):
        super().__init__()
        self.app = None
        self.dashboard = None
    
    def run(self) -> None:
        from PyQt5.QtWidgets import QApplication
        from market_monitor.gui.implementations.PyQt5Dashboard.builder import build_dashboard
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Trade Dashboard")
        self.logger.info("=" * 60)
        
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")
        
        self.dashboard = build_dashboard(self.config)
        self.dashboard.start()
        
        # Event loop Qt (bloccante)
        sys.exit(self.app.exec())


# =============================================================================
# PUBLIC API
# =============================================================================

def run_dashboard(
    config: Optional[Union[str, dict, Path]] = None,
    argv: Optional[List[str]] = None,
) -> None:
    """
    Avvia la dashboard PyQt5 di MarketMonitor.
    
    Args:
        config: Può essere:
            - None: usa argv o variabile d'ambiente
            - str: nome config, alias, o path al file YAML
            - Path: path al file YAML
            - dict: configurazione già caricata
        argv: Argomenti da linea di comando
    
    Examples:
        # Da CLI
        run_dashboard()
        
        # Con nome config
        run_dashboard("config_dashboard")
        
        # Con path diretto
        run_dashboard("/path/to/config.yaml")
        
        # Con dict
        run_dashboard({"name": "dashboard", ...})
    """
    runner = DashboardRunner()
    runner.execute(config=config, argv=argv)


def main() -> None:
    """Entry point per il comando CLI."""
    run_dashboard()


if __name__ == "__main__":
    main()
