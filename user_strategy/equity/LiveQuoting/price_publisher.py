import logging
from datetime import datetime
from typing import Dict, Optional, Protocol, Set, runtime_checkable

import numpy as np
import pandas as pd

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.publishers.timeseries_publisher import TimeSeriesPublisher

logger = logging.getLogger(__name__)


@runtime_checkable
class GuiPublisher(Protocol):
    """Protocol matching both RedisMessaging and RabbitMessaging."""

    def export_message(self, channel: str, value, **kwargs) -> None: ...
    def export_static_data(self, **data) -> None: ...


class PricePublisherHub:
    """Manages all price publishing: GUI channels (Redis/Rabbit) + TimeSeries storage."""

    _TS_FIELD_META: Dict[str, Dict[str, str]] = {
        'mid': {'type': 'MID'},
        'live_idx': {'type': 'MODEL_PRICE', 'model': 'index_cluster'},
        'live_clust': {'type': 'MODEL_PRICE', 'model': 'cluster'},
        'intraday': {'type': 'MODEL_PRICE', 'model': 'intraday_cluster'},
        'live_idx_mis': {'type': 'MISALIGNMENT', 'model': 'index_cluster'},
        'live_clust_mis': {'type': 'MISALIGNMENT', 'model': 'cluster'},
        'intraday_mis': {'type': 'MISALIGNMENT', 'model': 'intraday_cluster'},
    }

    _GUI_CHANNELS: Dict[str, str] = {
        'market:theoretical_live_index_cluster_price': 'live_idx',
        'market:theoretical_live_cluster_price': 'live_clust',
        'market:theoretical_live_intraday_price': 'intraday',
        'market:mid': 'mid',
        'market:normalized_mid': 'normalized_mid',
    }

    def __init__(self, gui: GuiPublisher,
                 timeseries: Optional[TimeSeriesPublisher],
                 isin_to_ticker: Dict[str, str]):
        self.gui = gui
        self.timeseries = timeseries
        self.isin_to_ticker = isin_to_ticker
        self.last_storage_time = datetime.now()

    @classmethod
    def from_config(cls, kwargs: dict, isin_to_ticker: Dict[str, str]) -> "PricePublisherHub":
        """Factory: build GUI publisher (Redis|Rabbit) + TimeSeries from config."""
        pub_cfg = kwargs.get("gui_publisher", {})
        pub_type = pub_cfg.get("type", "redis")

        if pub_type == "rabbit":
            from market_monitor.publishers.rabbit_publisher import RabbitMessaging
            rabbit_cfg = pub_cfg.get("rabbit", {})
            gui = RabbitMessaging(
                rabbit_host=rabbit_cfg.get("host", "rabbitmq.af.tst"),
                rabbit_port=rabbit_cfg.get("port", 5672),
                rabbit_user=rabbit_cfg.get("user", "mqclient"),
                rabbit_password=rabbit_cfg.get("password", "Mqclient-00"),
                rabbit_vhost=rabbit_cfg.get("vhost", "TestCredEQEtf"),
            )
            logger.info("GUI publisher: RabbitMQ")
        else:
            redis_cfg = pub_cfg.get("redis", {})
            gui = RedisMessaging(
                redis_host=redis_cfg.get("host", "localhost"),
                redis_port=redis_cfg.get("port", 6379),
                redis_db=redis_cfg.get("db", 0),
            )
            logger.info("GUI publisher: Redis")

        ts_cfg = kwargs.get("timeseries", {})
        try:
            timeseries = TimeSeriesPublisher(
                redis_host=ts_cfg.get("host", "localhost"),
                redis_port=ts_cfg.get("port", 6380),
                redis_db=ts_cfg.get("db", 0),
            )
        except Exception as e:
            timeseries = None
            logger.warning(f"Redis TS not connected: {e}")

        return cls(gui=gui, timeseries=timeseries, isin_to_ticker=isin_to_ticker)

    def publish_prices_to_gui(self, normalized_prices: Dict[str, pd.Series]) -> None:
        """Export normalized prices via pub/sub (Redis or RabbitMQ)."""
        for channel, price_key in self._GUI_CHANNELS.items():
            try:
                if (price := normalized_prices.get(price_key)) is not None:
                    self.gui.export_message(
                        channel,
                        price,
                        skip_if_unchanged=True,
                        flat_mode=True,
                    )
            except Exception as e:
                logger.info(f"export_message failed for {channel}: {e}", exc_info=True)

    def publish_returns(self, last_return: pd.Series, last_return_intraday: pd.Series) -> None:
        """Publish live return_0 and intraday_return_0."""
        self.gui.export_message("market:return_0",
                                (last_return.astype(float) * 100).round(4))
        self.gui.export_message("market:intraday_return_0",
                                (last_return_intraday.astype(float) * 100).round(4))

    def publish_static_returns(self, adjuster, periods: list) -> None:
        """Publish historical returns on market:return_1..N."""
        static_return = adjuster.get_clean_returns()
        for i in periods:
            self.gui.export_static_data(**{
                f"market:return_{i}": (static_return.iloc[-i].astype(float) * 100).round(4)
            })

    def publish_lf_data(self, intraday_adjuster) -> None:
        """Low-frequency update: publish intraday returns."""
        intraday_returns = intraday_adjuster.get_clean_returns()

        intraday_returns.index = intraday_returns.index.floor('min')
        intraday_returns.sort_index(ascending=False, inplace=True)
        intraday_returns.index = intraday_returns.index.strftime('%Y-%m-%dT%H:%M:%S')

        self.gui.export_static_data(df_big=intraday_returns.T.to_json())

    def publish_to_timeseries(self, normalized_prices: Dict[str, pd.Series],
                              current_time: datetime,
                              model_predictions: Dict[str, Optional[pd.Series]],
                              min_interval: float = 2.0) -> None:
        """Batch-publish to Redis TimeSeries if enough time has elapsed."""
        if self.timeseries is None:
            return
        if (current_time - self.last_storage_time).total_seconds() < min_interval:
            return

        all_isins: Set[str] = set()
        for series in model_predictions.values():
            if series is not None:
                all_isins.update(series.keys())

        if not all_isins:
            logger.warning("No ISINs found for TS storage")
            return

        mid_prices = normalized_prices['mid']
        price_fields = [
            ('intraday', normalized_prices['intraday']),
        ]

        count = 0
        try:
            with self.timeseries.ts_batch() as batch:
                for isin in all_isins:
                    mid_val = mid_prices.get(isin)
                    if not mid_val or mid_val == 0 or np.isnan(mid_val):
                        continue

                    batch.add(isin, 'mid', float(mid_val),
                              labels=self._build_ts_labels(isin, 'mid'))
                    count += 1

                    for field, series in price_fields:
                        val = series.get(isin)
                        if val is not None and not np.isnan(val):
                            batch.add(isin, field, float(val),
                                      labels=self._build_ts_labels(isin, field))
                            count += 1

        except Exception as e:
            logger.error(f"TS batch publishing failed: {e}", exc_info=True)
            raise

        self.last_storage_time = current_time
        logger.debug(f"TS storage published {count} values for {len(all_isins)} ISINs")

    def _build_ts_labels(self, isin: str, field: str) -> Dict[str, str]:
        """Build TS labels: ISIN, TICKER, type (MID|MODEL_PRICE|MISALIGNMENT), model."""
        labels: Dict[str, str] = {
            'isin': isin,
            'ticker': self.isin_to_ticker.get(isin, isin),
        }
        labels.update(self._TS_FIELD_META.get(field, {}))
        return labels

    @property
    def ts_stats(self) -> dict:
        """Return TimeSeries publisher statistics."""
        if self.timeseries is None:
            return {}
        return self.timeseries.ts_stats
