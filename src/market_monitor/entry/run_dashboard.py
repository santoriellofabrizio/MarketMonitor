"""
Entry point per la dashboard PyQt5.

Uso CLI:
    run-dashboard <config_name>
    run-dashboard --redis <channel>  [--host HOST] [--port PORT]
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
import argparse
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

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--redis",
            metavar="CHANNEL",
            default=None,
            help="Avvia in modalità Redis e sottoscrivi CHANNEL (es. trades_df)",
        )
        parser.add_argument(
            "--host",
            default="localhost",
            help="Host Redis (default: localhost, usato solo con --redis)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=6379,
            help="Porta Redis (default: 6379, usato solo con --redis)",
        )

    def execute(self, config=None, argv=None):
        """Override per gestire --redis prima della risoluzione config."""
        from market_monitor.entry._base import build_parser, setup_logging, handle_list_command

        parser = build_parser(
            description=self.description,
            extra_args=self.add_arguments,
        )
        self.args = parser.parse_args(argv)
        handle_list_command(self.args)

        log_level = self.args.log_level if hasattr(self.args, 'log_level') else "INFO"
        self.logger = setup_logging(
            level=log_level,
            log_file=self.log_file,
            name=f"market_monitor.{self.name}",
        )

        if self.args.redis:
            # Modalità Redis diretta — nessuna config YAML necessaria
            self.config = {
                "dashboard": {
                    "mode": "redis",
                    "redis": {
                        "host": self.args.host,
                        "port": self.args.port,
                        "channel": self.args.redis,
                    },
                }
            }
        else:
            from market_monitor.entry._base import resolve_config
            self.config = resolve_config(config, self.args, interactive=True)

        self.setup()
        try:
            self.run()
        except KeyboardInterrupt:
            print("\n[WARN]  Interruzione manuale.")
        except Exception as e:
            self.logger.exception(f"Errore fatale: {e}")
            raise
        finally:
            self.cleanup()

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

        # Modalità Redis diretta
        # run-dashboard --redis trades_df
        # run-dashboard --redis trades_df --host 192.168.1.10 --port 6380

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
