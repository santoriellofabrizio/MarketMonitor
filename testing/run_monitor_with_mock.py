import os
import sys
import warnings
from unittest.mock import patch

from ruamel.yaml import YAML

from testing.mock_bloomberg import MockBloombergStreamingThread
from testing.mock_market_trades_viewer import MockMarketTradesViewer

os.environ.setdefault('BBG_ROOT', 'xbbg')

reader = YAML(typ='safe')
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_monitor_with_mock(config=None, use_mock_bloomberg=True, use_mock_trades=True, mock_trades_config=None):
    if config is None:
        with open(sys.argv[1], 'r') as stream:
            config = reader.load(stream)

    if mock_trades_config is None:
        mock_trades_config = {
            "db_path": config.get("trade_distributor", {}).get("path", "mock_trades.db"),
            "trades_per_second": 0.5,
            "market": "ETFP",
        }

    mock_threads = []

    if use_mock_trades:
        mock_trade_viewer = MockMarketTradesViewer(**mock_trades_config)
        mock_threads.append(mock_trade_viewer)

    if use_mock_bloomberg:
        from market_monitor.builder import Builder
        with patch('market_monitor.builder.BloombergStreamingThread', MockBloombergStreamingThread):
            builder = Builder(config)
            threads, monitor = builder.build()
    else:
        from market_monitor.builder import Builder
        builder = Builder(config)
        threads, monitor = builder.build()

    all_threads = threads + mock_threads

    try:
        # Avvia threads
        for thread in mock_threads:
            thread.start()
            print(f"Started mock thread: {thread.name}")

        for thread in threads:
            thread.start()
            print(f"Started thread: {thread.name}")

        monitor.start()

        # ✅ CRITICO: Aspetta che il monitor finisca (altrimenti il main esce subito)
        print("\n[MONITOR] In esecuzione. Premi Ctrl+C per terminare.\n")
        monitor.join()  # Blocca qui fino a Ctrl+C o fino a quando monitor.stop() viene chiamato

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 [SHUTDOWN] Ctrl+C rilevato")
        print("=" * 60)

    finally:
        print("\n[SHUTDOWN] Chiusura in corso...")

        # 1. Stop tutti i thread
        for thread in all_threads:
            if hasattr(thread, 'stop'):
                print(f"  Stopping: {thread.name}")
                try:
                    thread.stop()
                except Exception as e:
                    print(f"  ⚠️ Errore: {e}")

        # 2. Stop monitor (chiama on_stop -> trade_manager.close)
        if hasattr(monitor, 'stop'):
            print(f"  Stopping monitor (questo chiama trade_manager.close)...")
            try:
                monitor.stop()  # Qui dentro c'è on_stop() -> trade_manager.close()
            except Exception as e:
                print(f"  ⚠️ Errore stop monitor: {e}")
                import traceback
                traceback.print_exc()

        # 3. Join threads con timeout
        print("\n[SHUTDOWN] Aspettando terminazione thread...")

        for thread in all_threads + [monitor]:
            if hasattr(thread, 'is_alive') and thread.is_alive():
                thread_name = getattr(thread, 'name', str(thread))
                print(f"  Joining {thread_name} (max 10s)...")
                thread.join(timeout=10.0)

                if thread.is_alive():
                    print(f"  ⚠️ {thread_name} ancora attivo!")
                else:
                    print(f"  ✓ {thread_name} terminato")

        print("\n" + "=" * 60)
        print("✅ [SHUTDOWN] Completato")
        print("=" * 60)


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