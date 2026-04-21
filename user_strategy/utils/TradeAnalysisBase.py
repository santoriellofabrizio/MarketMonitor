import logging
from abc import ABC, abstractmethod
from datetime import datetime
import datetime as dt

import pandas as pd

from market_monitor.publishers.rabbit_publisher import RabbitMessaging
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.common.trade_manager.book_memory import BookStorage
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

logger = logging.getLogger(__name__)


class TradeAnalysisBase(StrategyUI, ABC):
    """
    Template base for trade analysis strategies.

    Provides the full trade pipeline as a Template Method:
        on_trade → _enrich_trades → trade_manager → _post_trade_processing → publish

    Shared concrete behavior:
        - Dashboard messaging init (Redis + Rabbit)
        - publish_trades_on_dashboard
        - on_start_strategy  (calls _pre_start_setup hook, then publishes last trades)
        - update_HF          (time-gated call to get_live_data)
        - update_LF          (publish all trades)
        - on_stop            (close trade_manager)
        - instruments property

    Unified config keys (both engines must use this format):
        redis_data_export:
            activate: true
            params: {host: ..., port: ...}
            channel: "my_channel"
        rabbit_data_export:
            activate: true
            params: {host: ..., ...}
            channel: "my_channel"
    """

    HF_CUTOFF = dt.time(17, 29, 40)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.price_source: str = kwargs.get("price_source", "kafka")
        self.book_storage: BookStorage = BookStorage()
        self.trade_manager: TradeManager = TradeManager(
            self.book_storage, **kwargs.get("trade_manager", {})
        )

        self.redis_publisher: RedisMessaging | None = None
        self.redis_channel: str = ""
        self.rabbit_publisher: RabbitMessaging | None = None
        self.rabbit_channel: str = ""
        self._init_dashboard_messaging(kwargs)

    # ── Messaging setup ───────────────────────────────────────────────────────

    def _init_dashboard_messaging(self, kwargs: dict) -> None:
        redis_cfg = kwargs.get("redis_data_export", {})
        if redis_cfg.get("activate", False):
            self.redis_publisher = RedisMessaging(**redis_cfg.get("params", {}))
            self.redis_channel = redis_cfg.get("channel", "trades_redis")

        rabbit_cfg = kwargs.get("rabbit_data_export", {})
        if rabbit_cfg.get("activate", False):
            self.rabbit_publisher = RabbitMessaging(**rabbit_cfg.get("params", {}))
            self.rabbit_channel = rabbit_cfg.get("channel", "trades_rabbit")

    # ── Template methods ──────────────────────────────────────────────────────

    def on_start_strategy(self) -> None:
        self._pre_start_setup()
        last_trades = self.trade_manager.get_trades()
        if not last_trades.empty:
            self.publish_trades_on_dashboard(last_trades)

    def _pre_start_setup(self) -> None:
        """Hook called at the start of on_start_strategy, before publishing
        last trades. Override to perform engine-specific setup (e.g. populating
        instrument-to-market-id mappings)."""
        pass

    def update_HF(self) -> None:
        if datetime.today().time() < self.HF_CUTOFF:
            self.get_live_data()

    def update_LF(self) -> None:
        try:
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logger.error(e)

    def on_trade(self, new_trades: pd.DataFrame) -> None:
        new_trades = self._enrich_trades(new_trades)
        processed = self.trade_manager.on_trade(new_trades)
        self._post_trade_processing(processed)


    @staticmethod
    def _enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
        """Override to add domain-specific columns before trade_manager ingestion."""
        return trades

    def _post_trade_processing(self, processed: pd.DataFrame) -> None:
        """Hook called after trade_manager.on_trade(). Override for flow
        detection or other post-processing."""
        pass

    def publish_trades_on_dashboard(self, trades: pd.DataFrame) -> None:
        if self.redis_publisher:
            self.redis_publisher.export_message(
                channel=self.redis_channel,
                value=trades,
                date_format="iso",
                orient="records",
            )
        if self.rabbit_publisher:
            self.rabbit_publisher.export_message(
                channel=self.rabbit_channel,
                value=trades,
                date_format="iso",
                orient="records",
            )

    def on_stop(self) -> None:
        self.trade_manager.close()

    # ── Instruments property ──────────────────────────────────────────────────

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = value

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_live_data(self) -> None: ...

    @abstractmethod
    def on_market_data_setting(self) -> None: ...

    @abstractmethod
    def wait_for_book_initialization(self) -> bool: ...
