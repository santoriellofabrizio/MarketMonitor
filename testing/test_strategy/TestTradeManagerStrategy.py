import logging
from typing import Optional, Any

import pandas as pd

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.utils.trade_manager.book_memory import BookStorage
from user_strategy.utils.trade_manager.trade_manager import TradeManager

logger = logging.getLogger(__name__)


class TestTradeManagerStrategy(StrategyUI):
    """Tets
    Strategia che accumula e analizza i trades ricevuti.

    Test: Verifica che i trades si accumulino correttamente
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Strumenti da monitorare
        self.isins = [
            'IE0005042670',
            'LU0048584102',
            'IE0009470246',
            'IE0003400068',
            'LU0274211480',
            'IE0032174605',
            'LU0496736636',
            'LU0055732411',
            'IE0009505545',
            'LU0048584419',
            'IE0007266328',
            'DE0005933956',
            'LU0073263215',
            'IE0007266328',
            'LU0072462426',
            'LU0072462426',
            'LU0055732411',
            'IE0003400068',
            'LU0048584102',
            'DE0005327218',

        ]

        self.book_storage = BookStorage(maxlen=3)
        self.trade_manager = TradeManager(self.book_storage)
        self.redis_publisher = RedisMessaging()

    def on_config_change(self, key: str, old_value: Any, new_value: Any):
        print(key, old_value, new_value)

    def on_market_data_setting(self):
        """Setup market data."""
        subscription_dict = {
            isin: f"{isin} EQUITY"
            for isin in self.isins
        }
        self.market_data.subscription_dict_bloomberg = subscription_dict
        logger.info(f"Market data configured")

    def wait_for_book_initialization(self) -> bool:
        """Attendi book initialization."""
        mid = self.market_data.get_mid_eur()
        return len(mid[~mid.isna()]) >= 1

    def on_book_initialized(self):
        """Callback initialization."""
        logger.info("Book initialized!")

    def update_HF(self):
        """Update HF - monitora prezzi."""

        mid_eur = self.market_data.get_mid_eur()
        self.book_storage.append(mid_eur)

    def on_trade(self, trades: pd.DataFrame):
        """
        Accumula i trades ricevuti e computa statistiche.
        """

        self.trade_manager.on_trade(trades)
        trades = self.trade_manager.get_trades(10)
        self.redis_publisher.export_message(channel="trades_df", value=trades)

    def on_stop(self):
        self.trade_manager.close()

