"""
Testing package for MarketMonitorFI.

Contiene:
- Mock di bloomberg (simulatore dati di mercato)
- Mock di trade (simulatore trades in arrivo)
- Configurazioni di test e strategie dummy
- Test runner per orchestrare i test end-to-end

Uso:
    from testing.TestRunner import run_test
    from testing.TestConfig import TestConfig, test_strategy
    
    config = TestConfig(
        strategy=test_strategy.BASIC_DATA_FLOW,
        duration_seconds=30,
        trades_per_second=2.5
    )
    
    results = run_test(config)
"""

from .mock_bloomberg import MockBloombergStreamingThread
from .mock_market_trades_viewer import MockMarketTradesViewer

__all__ = [
    "MockBloombergStreamingThread",
    "MockMarketTradesViewer",
]
