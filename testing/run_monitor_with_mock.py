import os
import sys
import warnings
from unittest.mock import patch

from ruamel.yaml import YAML

from market_monitor.Builder import Builder
from testing import MockBloombergStreamingThread, MockMarketTradesViewer

os.environ.setdefault('BBG_ROOT', 'xbbg')

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_monitor_with_mock(config=None, use_mock_bloomberg=True, use_mock_trades=True, mock_trades_config=None):
    """
    Esegue il monitor con possibilità di usare i mock per Bloomberg e Trades.

    Args:
        config: Configurazione YAML (se None, legge da sys.argv[1])
        use_mock_bloomberg: Se True, usa MockBloombergStreamingThread
        use_mock_trades: Se True, lancia anche MockMarketTradesViewer
        mock_trades_config: Dizionario con configurazione per MockMarketTradesViewer
                           (db_path, trades_per_second, etf_instruments, etc.)
    """
    if config is None:
        with open(sys.argv[1], 'r') as stream:
            config = reader.load(stream)

    # Default config per mock trades
    if mock_trades_config is None:
        mock_trades_config = {
            "db_path": config.get("trade_distributor", {}).get("path", "mock_trades.db"),
            "trades_per_second": 2.5,
            "market": "ETFP",
        }

    # Thread aggiuntivi per i mock
    mock_threads = []

    # Setup mock trades viewer se richiesto
    if use_mock_trades:
        mock_trade_viewer = MockMarketTradesViewer(**mock_trades_config)
        mock_threads.append(mock_trade_viewer)

    # Patch Bloomberg se richiesto
    if use_mock_bloomberg:
        with patch('market_monitor.input_threads.bloomberg.BloombergStreamingThread', MockBloombergStreamingThread):
            builder = Builder(config)
            threads, monitor = builder.build()
    else:
        builder = Builder(config)
        threads, monitor = builder.build()

    # Aggiungi i mock threads
    all_threads = threads + mock_threads

    try:
        # Avvia prima i mock threads
        for thread in mock_threads:
            thread.start()
            print(f"Started mock thread: {thread.name}")

        # Poi i thread normali
        for thread in threads:
            thread.start()
            print(f"Started thread: {thread.name}")

        # Infine il monitor
        monitor.start()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

        # Stop in ordine inverso
        if hasattr(monitor, 'stop'):
            monitor.stop()

        for thread in all_threads:
            if hasattr(thread, 'stop'):
                thread.stop()

        # Aspetta che tutti i thread finiscano
        for thread in all_threads:
            if thread.is_alive():
                thread.join(timeout=2)


def run_monitor(config=None):
    """Wrapper standard che usa il sistema reale (compatibilità)"""
    run_monitor_with_mock(config, use_mock_bloomberg=False, use_mock_trades=False)


if __name__ == "__main__":
    # Parsing degli argomenti
    use_mock_bloomberg = '--mock-bloomberg' in sys.argv or '--mock' in sys.argv
    use_mock_trades = '--mock-trades' in sys.argv or '--mock' in sys.argv

    # Rimuovi i flag dagli argomenti
    for flag in ['--mock', '--mock-bloomberg', '--mock-trades']:
        if flag in sys.argv:
            sys.argv.remove(flag)

    # Configurazione custom per mock trades (opzionale)
    mock_trades_config = None

    # Esempio: leggi config da environment variable o usa default
    if os.getenv('MOCK_TRADES_PER_SEC'):
        mock_trades_config = {
            "trades_per_second": float(os.getenv('MOCK_TRADES_PER_SEC', 2.5)),
            "db_path": os.getenv('MOCK_TRADES_DB', "mock_trades.db"),
        }

    print("=" * 60)
    print("MARKET MONITOR - MOCK MODE")
    print("=" * 60)
    print(f"Mock Bloomberg: {'ENABLED' if use_mock_bloomberg else 'DISABLED'}")
    print(f"Mock Trades:    {'ENABLED' if use_mock_trades else 'DISABLED'}")
    print("=" * 60)

    run_monitor_with_mock(
        use_mock_bloomberg=True,
        use_mock_trades=True,
        mock_trades_config=mock_trades_config
    )