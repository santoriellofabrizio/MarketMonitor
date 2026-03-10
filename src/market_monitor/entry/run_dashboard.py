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
        from PyQt5.QtGui import QPalette, QColor
        from market_monitor.gui.implementations.PyQt5Dashboard.builder import build_dashboard

        self.logger.info("=" * 60)
        self.logger.info("Starting Trade Dashboard")
        self.logger.info("=" * 60)

        app_cfg = self.config.get("app", {}) if isinstance(self.config, dict) else {}
        qt_style = app_cfg.get("style", "Fusion")
        theme = app_cfg.get("theme", "light")

        self.logger.info(f"Qt style={qt_style!r}  theme={theme!r}")

        self.app = QApplication(sys.argv)
        self.app.setStyle(qt_style)

        if theme == "dark":
            self._apply_dark_palette(self.app)
            self.logger.info("Dark palette applied")

        self.dashboard = build_dashboard(self.config)
        self.dashboard.start()

        # Event loop Qt (bloccante)
        sys.exit(self.app.exec())

    @staticmethod
    def _apply_dark_palette(app) -> None:
        """Applica una palette scura all'applicazione Qt."""
        from PyQt5.QtGui import QPalette, QColor
        from PyQt5.QtCore import Qt

        palette = QPalette()
        dark = QColor(45, 45, 45)
        darker = QColor(30, 30, 30)
        mid = QColor(60, 60, 60)
        highlight = QColor(42, 130, 218)
        text = QColor(220, 220, 220)
        disabled_text = QColor(127, 127, 127)

        palette.setColor(QPalette.Window, dark)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, darker)
        palette.setColor(QPalette.AlternateBase, mid)
        palette.setColor(QPalette.ToolTipBase, dark)
        palette.setColor(QPalette.ToolTipText, text)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)
        palette.setColor(QPalette.Button, mid)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, highlight)
        palette.setColor(QPalette.Highlight, highlight)
        palette.setColor(QPalette.HighlightedText, Qt.black)

        app.setPalette(palette)


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
