import os
import sys
import warnings
from pathlib import Path
from unittest.mock import patch
from ruamel.yaml import YAML

# Import dai moduli del progetto (ora che sei nella root, questi funzionano)
from market_monitor.utils.config_helpers import (
    list_available,
    resolve_config_entry,
    suggest_names,
    ENV_DEFAULT_VAR,
    build_parser,  # Usiamo quello centralizzato
    print_config_details,
    load_config,
)

# --- SETUP AMBIENTE ---
# Ora siamo nella ROOT, quindi il parent è già la PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Aggiungiamo 'src' al path per trovare il pacchetto market_monitor
sys.path.insert(0, str(PROJECT_ROOT / "src"))

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def _get_mock_config(argv=None):
    """Gestisce il parsing includendo i flag di mock."""
    parser = build_parser()  # Assicurati che build_parser() in config_helpers accetti i flag mock

    # Se il build_parser standard non ha i mock, aggiungili qui:
    parser.add_argument("--mock", action="store_true", help="Enable all mocks.")
    parser.add_argument("--mock-bloomberg", action="store_true", help="Use simulated Bloomberg.")
    parser.add_argument("--mock-trades", action="store_true", help="Use simulated Trades.")

    args = parser.parse_args(argv)

    if args.list:
        list_available()
        sys.exit(0)

    config_name = args.config_name or os.environ.get(ENV_DEFAULT_VAR)
    if not config_name:
        print("❌ No config provided.")
        list_available()
        sys.exit(1)

    entry = resolve_config_entry(config_name)

    if args.describe:
        print_config_details(entry)
        sys.exit(0)

    config = load_config(entry.path)
    return config, args


def run_mock(config=None, argv=None):
    """Esecuzione con supporto Mock."""

    if config is None:
        config, args = _get_mock_config(argv)

    # Lazy Imports per velocità
    from market_monitor.builder import Builder
    from market_monitor.testing.mock_bloomberg import MockBloombergStreamingThread
    from market_monitor.testing.mock_market_trades_viewer import MockMarketTradesViewer

    # Setup Flags
    use_bbg_mock = args.mock or args.mock_bloomberg
    use_trades_mock = args.mock or args.mock_trades

    # 1. Inizializzazione Mock Trades (se richiesto)
    extra_threads = []
    if use_trades_mock:
        print("🛠️  Trades Mock ENABLED")
        mock_viewer = MockMarketTradesViewer(
            db_path=config["trade_distributor"]["path"],
            trades_per_second=0.5
        )
        extra_threads.append(mock_viewer)

    # 2. Build con Patch Bloomberg (se richiesto)
    if use_bbg_mock:
        print(" Bloomberg Mock ENABLED")
        with patch('market_monitor.builder.BloombergStreamingThread', MockBloombergStreamingThread):
            builder = Builder(config)
            threads, monitor = builder.build()
    else:
        builder = Builder(config)
        threads, monitor = builder.build()

    all_threads = threads + extra_threads

    # 3. Start & Lifecycle
    try:
        for t in all_threads: t.start()
        monitor.start()
        monitor.join()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        _cleanup(all_threads, monitor)


def _cleanup(threads, monitor):
    if monitor: monitor.stop()
    for t in threads:
        if hasattr(t, 'stop'): t.stop()
    for t in threads + ([monitor] if monitor else []):
        if t and t.is_alive(): t.join(timeout=5)
    print("✅ Done.")


if __name__ == "__main__":
    run_mock()