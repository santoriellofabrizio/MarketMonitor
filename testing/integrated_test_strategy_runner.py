"""
Test runner integrato che usa il Builder reale e mocka solo Bloomberg + trade.

Questo test:
1. Crea il Builder con config personalizzato
2. Usa MockBloombergStreamingThread per simulare prezzi
3. Usa MockMarketTradesViewer per simulare trades
4. Esegue la strategia completa
5. Raccoglie metriche
"""

import logging
import os
import time
import threading
from typing import Dict, Any, Optional

from market_monitor_fi.Builder import Builder
from market_monitor_fi.strategy.UserStrategy.StrategyRegister import register_strategy

from testing.mock_bloomberg import MockBloombergStreamingThread
from testing.mock_market_trades_viewer import MockMarketTradesViewer
from testing.test_strategy.FlowDetectingStrategy import FlowDetectingStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegratedTestRunner")


class IntegratedStrategyTestRunner:
    """
    Test runner che esegue una strategia completa con mock di Bloomberg e trade.
    
    Workflow:
    1. Prepara la configurazione
    2. Crea il Builder
    3. Avvia MockBloombergStreamingThread e MockMarketTradesViewer
    4. Esegue la strategia
    5. Raccoglie metriche
    """
    
    def __init__(self, 
                 strategy_name: str,
                 gui_type: str,
                 duration_seconds: int = 30,
                 num_etf_instruments: Optional[int] = None,
                 trades_per_second: float = 2.5,
                 market_update_interval: float = 0.5,
                 activate_trades: bool = True,
                 activate_bloomberg_mock: bool = True,
                 activate_redis: bool = False):
        """
        Inizializza il test runner.
        
        Args:
            strategy_name: Nome della strategia
            gui_type: Tipo di gui
            duration_seconds: Durata del test
            num_etf_instruments: Numero di ETF da simulare
            trades_per_second: Frequenza trades
            market_update_interval: Intervallo aggiornamenti Bloomberg
            activate_trades: Attiva trade_distributor e MockMarketTradesViewer
            activate_bloomberg_mock: Attiva MockBloombergStreamingThread
        """
        self.gui_type = gui_type
        self.trades_df_path = "testing"
        self.trades_db_name = "mock_trades.db"
        self.strategy_name = strategy_name
        self.duration_seconds = duration_seconds
        self.num_etf_instruments = num_etf_instruments
        self.trades_per_second = trades_per_second
        self.market_update_interval = market_update_interval
        
        # Config options
        self.config = {
            "activate_trades": activate_trades,
            "activate_bloomberg_mock": activate_bloomberg_mock,
            "activate_redis": activate_redis,
        }
        
        self.logger = logging.getLogger(f"TestRunner_{strategy_name}")
        
        # Componenti
        self.builder: Optional[Builder] = None
        self.strategy = None
        self.threads = []
        self.bloomberg_mock: Optional[MockBloombergStreamingThread] = None
        self.trade_mock: Optional[MockMarketTradesViewer] = None
        
        # Metriche
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "trades_received": 0,
            "market_updates": 0,
            "errors": [],
        }
    
    def setup(self) -> bool:
        """Setup del test - prepara la configurazione e il builder."""
        try:
            self.logger.info("="*70)
            self.logger.info("SETUP PHASE")
            self.logger.info("="*70)
            
            # 1. Crea configurazione
            config = self._create_config()
            self.logger.info(f"Config created for strategy: {self.strategy_name}")
            if self.config.get("activate_trades") and config.get("trade_distributor", {}).get("activate"):
                self._setup_market_trades_viewer()
            
            # 2. Crea builder
            self.builder = Builder(config)
            self.logger.info("Builder created")
            # 3. Build del sistema
            self.threads, self.strategy = self.builder.build()
            self.logger.info(f"Built {len(self.threads)} threads")
            
            # 4. Prepara i mock Bloomberg e trade
            self._setup_mocks()
            self.logger.info("Mocks setup completed")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            self.metrics["errors"].append(f"Setup error: {e}")
            return False
    
    def _create_config(self) -> Dict[str, Any]:
        """Crea la configurazione per il builder."""
        config = {
            "strategy_name": self.strategy_name,
            "logging": {
                "log_level": "INFO",
                "log_level_console": "INFO",
                "log_name": f"test_{self.strategy_name}.log",
            },
            "market_monitor_fi": {
                "tasks": {
                    "update_HF": {
                        "activate": True,
                        "frequency": self.market_update_interval,
                    },
                    "trade": {
                        "activate": self.config.get("activate_trades", True),
                        "synchronous": True,
                        "frequency": 0.1,
                    },
                    "update_LF": {
                        "activate": False,
                    },
                    "data_storing": {
                        "activate": False,
                    },
                    "dynamic_params": {
                        "activate": False,
                    },
                }
            },
            "market_data_distributor": {
                "activate": False,
                "book_params": {
                    "fields": ["BID", "ASK"],
                },
                "bloomberg_params": {},
            },

            "redis_data_distributor": {
                "activate": True,
                "redis_params": {"redis_host": "localhost", "redis_port": 6379, "redis_db": 0}
            },

            "trade_distributor": {
                "activate": self.config.get("activate_trades", True),
                "path": self.trades_df_path,
                "db_name": self.trades_db_name,
            },
            "gui": {
                "dummy": {
                    "gui_type": self.gui_type,
                },
            },
        }
        return config
    
    def _setup_mocks(self):
        """Configura i mock Bloomberg e trade."""
        # 1. MockBloombergStreamingThread (opzionale)
        if not self.config.get("activate_bloomberg_mock"):
            self.logger.info("Bloomberg mock disabled")
            return
            
        rtdata = self.strategy.market_data
        
        # Prepara gli ISIN dalla strategia
        if hasattr(self.strategy, 'isins'):
            isins = self.strategy.isins
        else:
            isins = MockMarketTradesViewer.DEFAULT_ETFS[:self.num_etf_instruments or 20]
            isins = [etf["isin"] for etf in isins]
        
        subscription_dict = {
            isin: f"{isin} EQUITY" 
            for isin in isins
        }
        
        # Crea mock event handler
        event_handler = type('MockEventHandler', (), {
            'RTData': rtdata,
            'get_securities_subscription_dict': lambda self: subscription_dict,
        })()
        
        self.bloomberg_mock = MockBloombergStreamingThread(
            event_processor=event_handler,
            subscription_dict=subscription_dict,
            update_interval=self.market_update_interval,
        )

    def _setup_market_trades_viewer(self, etf_list=None):

        etf_list = etf_list or [
            {"ticker": "XESC", "isin": "DE0007667107"},
            {"ticker": "VEUR", "isin": "LU0048584102"},
            {"ticker": "CSMM", "isin": "LU0072462426"},
            {"ticker": "AMND", "isin": "IE0002271879"},
            {"ticker": "LYXE", "isin": "LU0073263215"},
        ]

        if self.num_etf_instruments:
            etf_list = etf_list[:self.num_etf_instruments]

        self.trade_mock = MockMarketTradesViewer(
            db_path=os.path.join(self.trades_df_path, self.trades_db_name),
            trades_per_second=self.trades_per_second,
            etf_instruments=etf_list,
        )
    
    def run(self) -> bool:
        """Esegue il test."""
        try:
            if not self.setup():
                return False
            
            self.logger.info("="*70)
            self.logger.info("RUN PHASE")
            self.logger.info(f"Running for {self.duration_seconds} seconds")
            self.logger.info("="*70)
            
            self.metrics["start_time"] = time.time()
            
            # Avvia i thread
            for thread in self.threads:
                if not thread.is_alive():
                    thread.start()
                    self.logger.info(f"Started {thread.name}")
            
            # Avvia i mock
            if self.bloomberg_mock:
                self.bloomberg_mock.start()
                self.logger.info("Started MockBloombergStreamingThread")
            
            if self.trade_mock:
                self.trade_mock.start()
                self.logger.info("Started MockMarketTradesViewer")
            
            # Avvia la strategia (esegue asyncio.run)
            # NOTA: Questo è bloccante, quindi lo eseguiamo in un thread separato
            strategy_thread = threading.Thread(
                target=self.strategy.start,
                name="StrategyThread",
                daemon=False
            )
            strategy_thread.start()
            self.logger.info("Started StrategyUI")
            
            # Attendi la durata del test
            start = time.time()
            last_status = start
            
            while time.time() - start < self.duration_seconds:
                if time.time() - last_status >= 5:
                    self._log_status()
                    last_status = time.time()
                
                time.sleep(0.5)
            
            # Fine del test
            self.logger.info("Test duration completed, stopping...")
            self.metrics["end_time"] = time.time()
            self.metrics["duration"] = self.metrics["end_time"] - self.metrics["start_time"]
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error during run: {e}")
            self.metrics["errors"].append(f"Run error: {e}")
            return False
        
        finally:
            self.teardown()
    
    def _log_status(self):
        """Log dello status corrente."""
        elapsed = time.time() - self.metrics["start_time"]
        self.logger.info(f"Status at {elapsed:.1f}s: Still running...")
    
    def teardown(self):
        """Cleanup."""
        try:
            self.logger.info("="*70)
            self.logger.info("TEARDOWN PHASE")
            self.logger.info("="*70)
            
            # Stop dei mock
            if self.bloomberg_mock:
                self.bloomberg_mock.stop()
                self.bloomberg_mock.join(timeout=2)
                self.logger.info("Stopped MockBloombergStreamingThread")
            
            if self.trade_mock:
                self.trade_mock.stop()
                self.trade_mock.join(timeout=2)
                self.logger.info("Stopped MockMarketTradesViewer")
            
            # Stop della strategia
            if self.strategy:
                self.strategy.stop()
                self.logger.info("Stopped StrategyUI")
            
            # Stop dei thread
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=2)
                    self.logger.info(f"Stopped {thread.name}")
            
            self.logger.info("Teardown completed")
        
        except Exception as e:
            self.logger.error(f"Error during teardown: {e}")
            self.metrics["errors"].append(f"Teardown error: {e}")
    
    def print_results(self):
        """Stampa i risultati del test."""
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        
        duration = self.metrics["duration"] or 0
        print(f"\nStrategy: {self.strategy_name}")
        print(f"Duration: {duration:.1f} seconds")
        
        if self.metrics["errors"]:
            print(f"\nErrors ({len(self.metrics['errors'])}):")
            for error in self.metrics["errors"]:
                print(f"  - {error}")
        else:
            print("\nNo errors!")
        
        print("="*70 + "\n")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Ritorna le metriche del test."""
        return self.metrics


def run_integrated_test(strategy_name: str,
                        gui_type: str,
                        duration_seconds: int = 30,
                        **kwargs) -> Dict[str, Any]:
    """
    Funzione di convenienza per eseguire un test integrato.
    
    Args:
        strategy_name: Nome della strategia da testare
        duration_seconds: Durata del test
        **kwargs: Altri parametri (trades_per_second, market_update_interval, etc.)
    
    Returns:
        Dizionario con i risultati
    """
    runner = IntegratedStrategyTestRunner(
        strategy_name=strategy_name,
        duration_seconds=duration_seconds,
        gui_type=gui_type,
        **kwargs
    )
    
    success = runner.run()
    runner.print_results()
    
    return {
        "success": success,
        "metrics": runner.get_metrics(),
    }


if __name__ == "__main__":
    # Test della strategia SimplePriceMonitor
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║       INTEGRATED STRATEGY TEST - SimplePriceMonitor              ║")
    print("╚" + "="*68 + "╝")
    register_strategy("FlowDetectingStrategy", FlowDetectingStrategy)
    results = run_integrated_test(
        strategy_name="FlowDetectingStrategy",
        gui_type="RedisMessaging",
        duration_seconds=30,
        trades_per_second=2.5,
    )
    
    print(f"\nTest result: {'PASS' if results['success'] else 'FAIL'}")
