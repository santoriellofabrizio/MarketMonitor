import os
import sys
import warnings
from typing import Tuple, Optional, Any
from ruamel.yaml import YAML

# Importiamo solo le utilità di configurazione
from market_monitor.utils.config_helpers import (
    list_available,
    resolve_config_entry,
    suggest_names,
    ENV_DEFAULT_VAR,
    build_parser,
    print_config_details,
    load_config,
)

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def _setup_environment(argv: Optional[list]) -> Tuple[dict, Any]:
    """
    Gestisce il parsing, i comandi informativi e la risoluzione della config.
    Ritorna la config e gli argomenti (args) per l'esecuzione.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Comandi immediati (nessun file richiesto)
    if args.list:
        list_available()
        sys.exit(0)

    # 2. Risoluzione nome configurazione
    config_name = args.config_name or os.environ.get(ENV_DEFAULT_VAR)
    if not config_name:
        parser.print_usage()
        print(f"\nErrore: Nessuna configurazione. Imposta {ENV_DEFAULT_VAR} o passa un nome.")
        print("Opzioni disponibili:")
        list_available()
        sys.exit(1)

    # 3. Tentativo di caricamento
    try:
        entry = resolve_config_entry(config_name)
    except FileNotFoundError as e:
        print(f"{e}")
        if suggestions := suggest_names(config_name):
            print("\nForse intendevi:")
            for s in suggestions: print(f"  - {s}")
        sys.exit(1)

    # 4. Comandi informativi su config specifica
    if args.describe:
        print_config_details(entry)
        sys.exit(0)

    config = load_config(entry.path)

    if args.dry_run:
        print_config_details(entry)
        print("\n✅ Config validata con successo (Dry-run).")
        sys.exit(0)

    return config, args


def run_monitor(config: Optional[dict] = None, argv: Optional[list] = None):
    """Entry point principale del Monitor."""

    # Se richiamato da riga di comando, risolviamo l'ambiente
    if config is None:
        config, args = _setup_environment(argv)

    # --- Lazy Import ---
    from market_monitor.builder import Builder

    try:
        builder = Builder(config)
        threads, monitor = builder.build()

        # Avvio coerente
        for t in threads: t.start()
        monitor.start()

        print(f"\nStrategia attiva. [Config: {config.get('name', 'N/A')}]")
        monitor.join()

    except KeyboardInterrupt:
        print("\n Interruzione manuale rilevata.")
    except Exception as e:
        print(f"Errore fatale durante l'esecuzione: {e}")
    finally:
        _shutdown(threads if 'threads' in locals() else [],
                  monitor if 'monitor' in locals() else None)


def _shutdown(threads, monitor):
    """Logica di spegnimento pulita isolata."""
    print("\n[SHUTDOWN] Chiusura in corso...")
    if monitor and hasattr(monitor, 'stop'):
        monitor.stop()

    for t in threads:
        if hasattr(t, 'stop'): t.stop()

    for t in (threads + [monitor] if monitor else threads):
        if t and hasattr(t, 'is_alive') and t.is_alive():
            t.join(timeout=5.0)
    print("✅ Shutdown completato.")


if __name__ == "__main__":
    run_monitor()