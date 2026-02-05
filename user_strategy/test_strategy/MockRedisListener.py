"""
MockRedisListener - Test strategy that consumes data from RedisPublisher.

Subscribes to channels published by MockPricePublisher and processes data
from RTData stores via redis.
"""
import logging
from datetime import datetime
from time import sleep

import pandas as pd

from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

logger = logging.getLogger(__name__)


class MockRedisListener(StrategyUI):
    """
    Test strategy that listens to RedisPublisher channels and accesses RTData stores.
    
    Subscribes to:
        - market:equity:prices (MarketStore)
        - market:fx:rates (MarketStore)
        - portfolio:pnl:realtime (StateStore)
        - metadata:strategy_status (StateStore)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.instruments = kwargs.get("instruments", [
            "DE0007667107",
            "LU0048584102",
            "LU0072462426",
        ])
        
        self.nav = pd.Series(index=self.instruments, dtype=float)
        self.mid = pd.Series(index=self.instruments, dtype=float)
        
        logger.info(f"MockRedisListener initialized for {len(self.instruments)} instruments")

    def on_market_data_setting(self) -> None:
        """Subscribe to RedisPublisher channels."""
        # Subscribe to market channels (same as EtfEquityLiveAnalysis)
        for channel in ["nav", "book"]:
            self.market_data.subscribe_redis(
                channel=f"market:{channel}",
                store="market"
            )
        
        logger.info("Subscribed to RedisPublisher channels")

    def wait_for_book_initialization(self) -> bool:
        """Wait for initial data from RedisPublisher."""
        logger.info("Waiting for RedisPublisher data...")
        timeout = 10
        start = datetime.now()
        
        while (datetime.now() - start).seconds < timeout:
            # Check if we have data from MarketStore
            try:
                mid = self.market_data.get_data_field(field="MID", index_data="market")
                if mid is not None and not mid.empty:
                    logger.info("Initial data received from RedisPublisher")
                    return True
            except:
                pass
            
            sleep(0.5)
        
        logger.warning("Timeout waiting for RedisPublisher data")
        return True  # Continue anyway for testing

    def on_book_initialized(self):
        """Called once initial data is available."""
        logger.info("Book initialized with RedisPublisher data")

    def update_HF(self):
        """High-frequency update - process data from RTData stores."""
        # Read from MarketStore
        self.nav.update(self.market_data.get_data_field(field="nav", index_data="market") or {})
        self.mid.update(self.market_data.get_data_field(field="book", index_data="market") or {})
        
        # Log summary
        if not self.nav.empty:
            logger.info(f"Received {len(self.nav.dropna())} NAV prices, {len(self.mid.dropna())} book prices")

    def update_LF(self):
        """Low-frequency update - log aggregated data."""
        if not self.nav.empty:
            logger.info(f"\n--- Summary ---")
            logger.info(f"NAV prices: {self.nav.dropna().to_dict()}")
            logger.info(f"Book prices: {self.mid.dropna().to_dict()}")

    def on_stop(self):
        """Cleanup."""
        logger.info("MockRedisListener stopped")
