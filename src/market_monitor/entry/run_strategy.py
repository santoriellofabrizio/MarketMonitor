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


class StrategyRunner(BaseRunner):
    """Runner per l'esecuzione delle strategie."""
    
    name = "strategy"
    description = "Esegue una user strategy su MarketMonitor"
    
    def __init__(self):
        super().__init__()
        self.threads: List = []
        self.monitor = None
    
    def run(self) -> None:
        from market_monitor.builder import Builder
        
        builder = Builder(self.config)
        self.threads, self.monitor = builder.build()
        
        # Start threads
        for t in self.threads:
            t.start()
        
        self.monitor.start()
        
        config_name = self.config.get('name', 'N/A')
        print(f"\n✅ Strategia attiva. [Config: {config_name}]")
        
        # Wait for monitor
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
