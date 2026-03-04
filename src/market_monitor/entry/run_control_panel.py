"""
Entry point per il StrategyControlPanel standalone.

Avvia il control panel come processo separato: si connette a Redis e
interagisce con qualsiasi strategia già in esecuzione (in un altro processo).
Può anche lanciare la strategia direttamente dal pannello tramite il pulsante
"Start Strategy" (che usa QProcess internamente).

Uso CLI:
    run-control-panel
    run-control-panel --config my_strategy_config
    run-control-panel --host localhost --port 6379 --db 0
    run-control-panel --lifecycle-channel engine:lifecycle
    run-control-panel --title "My Strategy"

Uso programmatico:
    from market_monitor.entry.run_control_panel import launch_control_panel
    launch_control_panel(host="localhost", port=6379)
"""
import argparse
import sys


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="run-control-panel",
        description="Avvia il StrategyControlPanel standalone.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Preseleziona un config nella lista (es. my_strategy_config)",
    )
    parser.add_argument("--host",              default="localhost",       help="Redis host (default: localhost)")
    parser.add_argument("--port",              type=int, default=6379,    help="Redis port (default: 6379)")
    parser.add_argument("--db",                type=int, default=0,       help="Redis DB (default: 0)")
    parser.add_argument("--commands-channel",  default="engine:commands", help="Canale comandi Redis")
    parser.add_argument("--status-channel",    default="engine:status",   help="Canale risposte Redis")
    parser.add_argument("--lifecycle-channel", default="engine:lifecycle", help="Canale eventi lifecycle Redis")
    parser.add_argument("--title",             default="Strategy Control Panel", help="Titolo della finestra")
    return parser.parse_args(argv)


def launch_control_panel(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    commands_channel: str = "engine:commands",
    status_channel: str = "engine:status",
    lifecycle_channel: str = "engine:lifecycle",
    initial_config: str = None,
    title: str = "Strategy Control Panel",
) -> None:
    """
    Avvia il StrategyControlPanel in modalità standalone.

    Crea una QApplication, istanzia il pannello, lo mostra e avvia il Qt event loop.
    Blocca finché l'utente non chiude la finestra.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database index
        commands_channel: Canale Redis su cui pubblicare i comandi
        status_channel: Canale Redis da cui ricevere le risposte ai comandi
        lifecycle_channel: Canale Redis da cui ricevere gli eventi lifecycle
        initial_config: Config pre-selezionato nel dropdown (opzionale)
        title: Titolo della finestra
    """
    from PyQt5.QtWidgets import QApplication
    from market_monitor.gui.implementations.StrategyControlPanel import StrategyControlPanel

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    panel = StrategyControlPanel(
        redis_config={"host": host, "port": port, "db": db},
        commands_channel=commands_channel,
        status_channel=status_channel,
        lifecycle_channel=lifecycle_channel,
        initial_config=initial_config,
    )
    panel.setWindowTitle(title)
    panel.show()

    sys.exit(app.exec_())


def main() -> None:
    """Entry point per il comando CLI run-control-panel."""
    args = _parse_args()
    launch_control_panel(
        host=args.host,
        port=args.port,
        db=args.db,
        commands_channel=args.commands_channel,
        status_channel=args.status_channel,
        lifecycle_channel=args.lifecycle_channel,
        initial_config=args.config,
        title=args.title,
    )


if __name__ == "__main__":
    main()
