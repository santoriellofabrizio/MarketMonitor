"""
Entry point per l'esecuzione della strategia.

Uso CLI:
    run-strategy <config_name>
    run-strategy --list
    run-strategy --describe <config_name>
    run-strategy --dry-run <config_name>

Uso programmatico:
    from market_monitor.entry.run_strategy import run_strategy

    run_strategy("config_name")
    run_strategy("/path/to/config.yaml")
    run_strategy({"name": "test", ...})
"""
from typing import List, Optional, Union
from pathlib import Path

from market_monitor.entry._base import BaseRunner, shutdown_threads


# GUI types that require a Qt event loop in the main thread
_QT_GUI_TYPES = {"StrategyControlPanel"}


def _has_qt_gui(config: dict) -> bool:
    """Return True if any GUI in the config requires a Qt event loop."""
    return any(
        p.get("gui_type") in _QT_GUI_TYPES
        for p in config.get("gui", {}).values()
    )


class StrategyRunner(BaseRunner):
    """Runner per l'esecuzione delle strategie."""

    name = "strategy"
    description = "Esegue una user strategy su MarketMonitor"

    def __init__(self):
        super().__init__()
        self.threads: List = []
        self.monitor = None
        self._qt_app = None

    def run(self) -> None:
        from market_monitor.builder import Builder

        # If any GUI requires Qt, create QApplication BEFORE builder.build()
        # so that QMainWindow can be instantiated safely.
        use_qt = _has_qt_gui(self.config)
        if use_qt:
            import sys
            from PyQt5.QtWidgets import QApplication
            self._qt_app = QApplication.instance() or QApplication(sys.argv)
            self._qt_app.setStyle("Fusion")

        builder = Builder(self.config)
        self.threads, self.monitor = builder.build()

        for t in self.threads:
            t.start()

        if use_qt:
            import sys
            import threading

            # Run the asyncio strategy loop in a background thread so that
            # the main thread is free to run the Qt event loop.
            strategy_thread = threading.Thread(
                target=self.monitor.start,
                daemon=True,
                name="StrategyThread",
            )
            strategy_thread.start()

            # Show all Qt windows registered as GUIs
            for gui in self.monitor.GUIs.values():
                if hasattr(gui, "show"):
                    gui.show()

            config_name = self.config.get("name", "N/A")
            print(f"\n✅ Strategia attiva. [Config: {config_name}]")

            # Qt event loop blocks the main thread until all windows are closed
            sys.exit(self._qt_app.exec_())
        else:
            # Original behaviour: asyncio loop runs in the main thread
            self.monitor.start()

            config_name = self.config.get("name", "N/A")
            print(f"\n✅ Strategia attiva. [Config: {config_name}]")

            self.monitor.join()

    def cleanup(self) -> None:
        shutdown_threads(self.threads, self.monitor)


# =============================================================================
# PUBLIC API
# =============================================================================

_runner = StrategyRunner()


def run_strategy(
    config: Optional[Union[str, dict, Path]] = None,
    argv: Optional[List[str]] = None,
) -> None:
    """
    Esegue una strategia MarketMonitor.

    Args:
        config: Può essere:
            - None: usa argv o variabile d'ambiente
            - str: nome config, alias, o path al file YAML
            - Path: path al file YAML
            - dict: configurazione già caricata
        argv: Argomenti da linea di comando (se config è None)

    Examples:
        # Da CLI
        run_strategy()

        # Con nome config
        run_strategy("my_strategy")

        # Con path diretto
        run_strategy("/path/to/config.yaml")

        # Con dict
        run_strategy({"name": "test", ...})
    """
    runner = StrategyRunner()
    runner.execute(config=config, argv=argv)


def main() -> None:
    """Entry point per il comando CLI."""
    run_strategy()


if __name__ == "__main__":
    main()
