"""
Entry point per l'esecuzione con mock (Bloomberg/Trades simulati).

Uso CLI:
    run-mock <config_name>
    run-mock <config_name> --mock              # Abilita tutti i mock
    run-mock <config_name> --mock-bloomberg    # Solo mock Bloomberg
    run-mock <config_name> --mock-trades       # Solo mock Trades
    run-mock --list
    run-mock --describe <config_name>

Uso programmatico:
    from market_monitor.entry.run_mock import run_mock
    
    run_mock("config_name", mock_bloomberg=True)
    run_mock("/path/to/config.yaml", mock_all=True)
"""
import argparse
from typing import List, Optional, Union
from pathlib import Path
from unittest.mock import patch

from market_monitor.entry._base import BaseRunner, shutdown_threads


class MockRunner(BaseRunner):
    """Runner per l'esecuzione con mock."""
    
    name = "mock"
    description = "Esegue MarketMonitor con dati simulati (mock)"
    
    def __init__(
        self,
        mock_bloomberg: bool = False,
        mock_trades: bool = False,
    ):
        super().__init__()
        self.mock_bloomberg = mock_bloomberg
        self.mock_trades = mock_trades
        self.threads: List = []
        self.monitor = None
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mock",
            action="store_true",
            help="Abilita tutti i mock (Bloomberg + Trades)",
        )
        parser.add_argument(
            "--mock-bloomberg",
            action="store_true",
            help="Usa Bloomberg simulato",
        )
        parser.add_argument(
            "--mock-trades",
            action="store_true",
            help="Usa Trades simulati",
        )
    
    def setup(self) -> None:
        # Aggiorna flags dai args CLI
        if self.args:
            if self.args.mock:
                self.mock_bloomberg = True
                self.mock_trades = True
            if self.args.mock_bloomberg:
                self.mock_bloomberg = True
            if self.args.mock_trades:
                self.mock_trades = True
    
    def run(self) -> None:
        from market_monitor.builder import Builder
        from market_monitor.testing.mock_bloomberg import MockBloombergStreamingThread
        from market_monitor.testing.mock_market_trades_viewer import MockMarketTradesViewer
        
        extra_threads = []
        
        # Setup mock trades
        if self.mock_trades:
            print("🛠️  Trades Mock ENABLED")
            mock_viewer = MockMarketTradesViewer(
                db_path=self.config["trade_distributor"]["path"],
                trades_per_second=0.5,
            )
            extra_threads.append(mock_viewer)
        
        # Build con eventuale mock Bloomberg
        if self.mock_bloomberg:
            print("🛠️  Bloomberg Mock ENABLED")
            with patch(
                'market_monitor.builder.BloombergStreamingThread',
                MockBloombergStreamingThread,
            ):
                builder = Builder(self.config)
                self.threads, self.monitor = builder.build()
        else:
            builder = Builder(self.config)
            self.threads, self.monitor = builder.build()
        
        self.threads.extend(extra_threads)
        
        # Start
        for t in self.threads:
            t.start()
        
        self.monitor.start()
        
        config_name = self.config.get('name', 'N/A')
        print(f"\n✅ Mock attivo. [Config: {config_name}]")
        
        self.monitor.join()
    
    def cleanup(self) -> None:
        shutdown_threads(self.threads, self.monitor)


# =============================================================================
# PUBLIC API
# =============================================================================

def run_mock(
    config: Optional[Union[str, dict, Path]] = None,
    argv: Optional[List[str]] = None,
    mock_all: bool = False,
    mock_bloomberg: bool = False,
    mock_trades: bool = False,
) -> None:
    """
    Esegue MarketMonitor con dati simulati.
    
    Args:
        config: Può essere:
            - None: usa argv o variabile d'ambiente
            - str: nome config, alias, o path al file YAML
            - Path: path al file YAML
            - dict: configurazione già caricata
        argv: Argomenti da linea di comando
        mock_all: Abilita tutti i mock
        mock_bloomberg: Abilita mock Bloomberg
        mock_trades: Abilita mock Trades
    
    Examples:
        # Da CLI con tutti i mock
        run_mock()  # usa --mock da CLI
        
        # Programmatico con mock specifici
        run_mock("my_config", mock_bloomberg=True)
        
        # Tutti i mock
        run_mock("my_config", mock_all=True)
    """
    if mock_all:
        mock_bloomberg = True
        mock_trades = True
    
    runner = MockRunner(
        mock_bloomberg=mock_bloomberg,
        mock_trades=mock_trades,
    )
    runner.execute(config=config, argv=argv)


def main() -> None:
    """Entry point per il comando CLI."""
    run_mock()


if __name__ == "__main__":
    main()
