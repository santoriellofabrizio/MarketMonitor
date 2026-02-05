"""
MockPricePublisher - Test strategy that publishes mock prices to RedisPublisher.

Integrates with IntegratedStrategyTestRunner infrastructure.
"""
import logging
from datetime import datetime

import pandas as pd

from market_monitor.publishers.base import MessageType
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

logger = logging.getLogger(__name__)


class MockPricePublisher(StrategyUI):
    """
    Test strategy that publishes mock Bloomberg prices to RedisPublisher.
    Simulates EtfEquityPriceEngine behavior for testing store routing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.gui = RedisMessaging()

        # Mock instruments (from test_strategy defaults if available)
        self.instruments = kwargs.get("instruments", [
            "DE0007667107",  # XESC
            "LU0048584102",  # VEUR
            "LU0072462426",  # CSMM
        ])
        
        self.currencies = ["USD", "EUR", "GBP"]
        self.all_securities = self.instruments + self.currencies
        
        # Base prices
        self.base_prices = {isin: 100.0 + i*10 for i, isin in enumerate(self.instruments)}
        self.base_prices.update({"USD": 1.10, "EUR": 1.0, "GBP": 0.85})
        
        # Mock positions
        self.positions = {
            self.instruments[0]: {"qty": 100, "avg_price": self.base_prices[self.instruments[0]] * 0.95},
        }
        
        logger.info(f"MockPricePublisher initialized with {len(self.instruments)} instruments")

    def on_market_data_setting(self) -> None:
        """Setup market data securities and subscriptions."""
        self.market_data.set_securities(self.all_securities)
        
        # Set currency info
        currency_info = {isin: "EUR" for isin in self.instruments}
        self.market_data.currency_information = currency_info
        
        # Subscribe to Bloomberg mock
        for isin in self.all_securities:
            self.market_data.subscribe_bloomberg(
                id=isin,
                subscription_string=f"{isin} EQUITY",
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def wait_for_book_initialization(self) -> bool:
        """Initialize book with base prices."""
        for isin, price in self.base_prices.items():
            if isin in self.all_securities:
                spread = price * 0.001
                self.market_data.update(isin, {
                    "BID": price - spread / 2,
                    "ASK": price + spread / 2
                })
        
        logger.info("Book initialized")
        return True

    def on_book_initialized(self):
        """Called once book is initialized."""
        # Publish initial portfolio state
        self.gui.export_message("portfolio:positions", self.positions)
        
        # Initial PnL
        pnl = self._calculate_pnl()
        self.gui.export_message("portfolio:pnl", pnl)
        
        logger.info("Initial state published to RedisPublisher")

    def update_HF(self):
        """High-frequency update - publishes to RedisPublisher on every market update."""
        mid_eur = self.market_data.get_mid_eur()
        
        # Publish NAV-like data using export_data_message
        self.gui.export_message("market:nav", mid_eur, skip_if_unchanged=True)
        self.gui.export_message("market:book", mid_eur,skip_if_unchanged=True)

    def _calculate_pnl(self):
        """Calculate PnL from positions."""
        mid_eur = self.market_data.get_mid_eur()
        pnl = {}
        
        for isin, pos in self.positions.items():
            if isin in mid_eur.index and not pd.isna(mid_eur[isin]):
                current_price = float(mid_eur[isin])
                pnl[isin] = (current_price - pos["avg_price"]) * pos["qty"]
        
        if pnl:
            pnl["total"] = sum(pnl.values())
        
        return pnl

    def update_LF(self):
        """Low-frequency update."""
        pass

    def on_stop(self):
        """Cleanup."""
        self.gui.export_message("metadata:strategy_status", {
            "name": "MockPricePublisher",
            "status": "stopped",
            "last_update": datetime.now().isoformat()
        }, MessageType.STATUS)
        logger.info("MockPricePublisher stopped")
