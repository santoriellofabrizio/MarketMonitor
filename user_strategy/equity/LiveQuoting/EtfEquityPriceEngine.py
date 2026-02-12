import logging
import os
import sqlite3
import time as sleep_time
from collections import deque
from datetime import datetime, time, date
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from dateutil.utils import today

from sfm_data_provider.analytics.adjustments.adjuster import Adjuster
from sfm_data_provider.analytics.adjustments.dividend import DividendComponent
from sfm_data_provider.analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from sfm_data_provider.analytics.adjustments.fx_spot import FxSpotComponent
from sfm_data_provider.analytics.adjustments.ter import TerComponent
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.gui.implementations.GUI import GUI
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.publishers.timeseries_publisher import TimeSeriesPublisher
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

from user_strategy.equity.LiveQuoting.InputParamsQuoting import InputParamsQuoting
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager
from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator, TrimmedMean
from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel

logger = logging.getLogger(__name__)


class EtfEquityPriceEngine(StrategyUI):

    # Labels per ogni campo TimeSeries: type (MID | MODEL_PRICE | MISALIGNMENT) e model opzionale
    _TS_FIELD_META: Dict[str, Dict[str, str]] = {
        'mid':            {'type': 'MID'},
        'live_idx':       {'type': 'MODEL_PRICE', 'model': 'index_cluster'},
        'live_clust':     {'type': 'MODEL_PRICE', 'model': 'cluster'},
        'intraday':       {'type': 'MODEL_PRICE', 'model': 'intraday_cluster'},
        'live_idx_mis':   {'type': 'MISALIGNMENT', 'model': 'index_cluster'},
        'live_clust_mis': {'type': 'MISALIGNMENT', 'model': 'cluster'},
        'intraday_mis':   {'type': 'MISALIGNMENT', 'model': 'intraday_cluster'},
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_publisher(kwargs)
        self._init_universe(kwargs)
        self._init_historical_data(kwargs)
        self._init_bloomberg(kwargs)

    # =========================================================================
    # Inizializzazione
    # =========================================================================

    def _init_publisher(self, kwargs: dict) -> None:
        """Inizializza gui_redis (Redis o RabbitMQ) e timeseries_publisher."""
        pub_cfg = kwargs.get("gui_publisher", {})
        pub_type = pub_cfg.get("type", "redis")

        if pub_type == "rabbit":
            from market_monitor.publishers.rabbit_publisher import RabbitMessaging
            rabbit_cfg = pub_cfg.get("rabbit", {})
            self.gui_redis = RabbitMessaging(
                rabbit_host=rabbit_cfg.get("host", "rabbitmq.af.tst"),
                rabbit_port=rabbit_cfg.get("port", 5672),
                rabbit_user=rabbit_cfg.get("user", "mqclient"),
                rabbit_password=rabbit_cfg.get("password", "Mqclient-00"),
                rabbit_vhost=rabbit_cfg.get("vhost", "TestCredEQEtf"),
            )
            logger.info("GUI publisher: RabbitMQ")
        else:
            redis_cfg = pub_cfg.get("redis", {})
            self.gui_redis = RedisMessaging(
                redis_host=redis_cfg.get("host", "localhost"),
                redis_port=redis_cfg.get("port", 6379),
                redis_db=redis_cfg.get("db", 0),
            )
            logger.info("GUI publisher: Redis")

        ts_cfg = kwargs.get("timeseries", {})
        try:
            self.timeseries_publisher = TimeSeriesPublisher(
                redis_host=ts_cfg.get("host", "localhost"),
                redis_port=ts_cfg.get("port", 6380),
                redis_db=ts_cfg.get("db", 0),
            )
        except Exception as e:
            self.timeseries_publisher = None
            self.logger.warning(f"Redis TS not connected: {e}")

    def _init_universe(self, kwargs: dict) -> None:
        """Carica l'universo ETF, le valute e i mapping ISIN/ticker."""
        self.number_of_days = kwargs.get("number_of_days", 5)
        self.number_of_days_intraday = kwargs.get("number_of_days_intraday", 3)
        self.failed_isin = []

        self.API = BshData(config_path=kwargs.get("bshdata_config_path"))

        isins_etf_equity = self.API.general.get(
            fields=["etp_isins"], segments=["IM"], currency="EUR",
            underlying="EQUITY", source="oracle",
        )["etp_isins"]

        self.reference_tick_size = self.API.info.get_etp_fields(
            isin=isins_etf_equity, fields=["REFERENCE_TICK_SIZE"], source="bloomberg",
        )["REFERENCE_TICK_SIZE"].to_dict()

        db_path = kwargs["path_db"]
        with sqlite3.connect(db_path) as conn:
            self.isin_to_ticker = pd.read_sql(
                "SELECT ISIN, TICKER FROM vw_isin_ticker", conn
            ).set_index("ISIN")["TICKER"].to_dict()
            self.active_isin = pd.read_sql(
                "SELECT ISIN, MARKET_STATUS FROM market_status", conn
            ).set_index("ISIN")["MARKET_STATUS"].to_dict()

        self.currencies = [
            f"EUR{c}" for c in
            ['USD', 'GBP', 'CHF', 'AUD', 'DKK', 'HKD', 'NOK',
             'PLN', 'SEK', 'CNY', 'JPY', 'CNH', 'CAD', 'INR', 'BRL']
        ]

        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        self.input_params = InputParamsQuoting(**kwargs)

        # ETF universe: unione degli indici/colonne delle matrici beta
        raw_etfs = list({
            *self.input_params.beta_cluster.index,
            *self.input_params.beta_cluster.columns,
            *self.input_params.beta_cluster_index.index,
            *self.input_params.beta_cluster_index.columns,
        })
        isin_list = kwargs.get("isin_list", [])
        valid_etfs = []
        for etf in raw_etfs:
            if isin_list and etf not in isin_list:
                continue
            if self.active_isin.get(etf, "NOT_PRESENT") != "ACTV":
                logger.warning(f"{etf} ({self.ticker_to_isin.get(etf)}) is not ACTV.")
                continue
            valid_etfs.append(etf)
        self.etfs = sorted(valid_etfs)

        # Rifiltra e rinormalizza le matrici beta sugli ETF rimasti
        for attr in ['beta_cluster', 'beta_cluster_index']:
            df = getattr(self.input_params, attr)
            df_filtered = df.loc[
                df.index.intersection(self.etfs),
                df.columns.intersection(self.etfs)
            ]
            setattr(self.input_params, attr,
                    df_filtered.div(df_filtered.sum(axis=1), axis=0).fillna(0))

        self.instruments = list({*self.etfs, *self.currencies})
        self._isins_etf_equity = list(isins_etf_equity)

        # Stato runtime
        self.mid_eur: Optional[pd.Series] = None
        self.book_mid_threshold = 0.5
        self.book_eur: pd.DataFrame = pd.DataFrame()
        self.book_storage: deque = deque(maxlen=3)
        self.position: Optional[pd.Series] = None
        self.return_to_publish: list = [1, 2, 3, 4, 5, 6, 7, 8]
        self.today = pd.Timestamp.today().normalize()
        self.yesterday = HolidayManager().previous_business_day(self.today)

    def _init_historical_data(self, kwargs: dict) -> None:
        """Carica prezzi storici, adjusters e modelli di pricing."""
        self.holidays = HolidayManager()
        start = self.holidays.subtract_business_days(today(), self.number_of_days)
        start_intraday = self.holidays.subtract_business_days(today(), self.number_of_days_intraday)
        end = self.holidays.previous_business_day(today())
        days = self.holidays.get_business_days(start=start, end=end)
        snapshot_time = time(16, 45)
        self.last_storage_time = datetime.now()

        fx_composition = self.API.info.get_fx_composition(
            self.etfs, fx_fxfwrd="fx", reference_date=date(2026, 2, 9))
        fx_forward = self.API.info.get_fx_composition(
            self.etfs, fx_fxfwrd="fxfwrd", reference_date=date(2026, 2, 9))

        for isin, isin_proxy in kwargs.get("fx_mapping", {}).items():
            fx_composition.loc[isin] = fx_composition.loc[isin_proxy]

        self.fx_prices = self.API.market.get_daily_currency(
            id=self.currencies, start=start, end=end, snapshot_time=snapshot_time,
            fallbacks=[{"source": "bloomberg"}],
        ).reindex(days)

        self.etf_prices = self.API.market.get_daily_etf(
            id=self.etfs, start=start, end=end, snapshot_time=snapshot_time, timeout=10,
            fallbacks=[{"source": "bloomberg", "market": mkt} for mkt in ["IM", "FP", "NA"]],
        ).reindex(days)

        self.fx_prices_intraday = self.filter_outliers(
            self.API.market.get_intraday_fx(
                id=self.currencies, start=start_intraday, end=end, frequency="15m",
                fallbacks=[{"source": "bloomberg"}])
            .between_time("10:00", "17:00"))

        self.etf_prices_intraday = self.filter_outliers(
            self.API.market.get_intraday_etf(
                id=self.etfs, start=start_intraday, end=end, frequency="15m",
                source='timescale',
                fallbacks=[{"source": "bloomberg", "market": mkt} for mkt in ["IM", "FP", "NA"]])
            .between_time("10:00", "17:00"), name="etf_intraday")

        self.etf_prices = self.etf_prices.interpolate("time")
        self.etf_prices_intraday = self.etf_prices_intraday.interpolate("time")

        fx_forward_prices = self.API.market.get_daily_fx_forward(
            quoted_currency=fx_forward.columns.tolist(), start=start, end=end)
        dividends = self.API.info.get_dividends(id=self.etfs, start=start)
        ter = self.API.info.get_ter(id=self.etfs) / 100

        self.adjuster = (
            Adjuster(self.etf_prices)
            .add(TerComponent(ter))
            .add(FxSpotComponent(fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(fx_forward, fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(dividends, self.etf_prices, fx_prices=self.fx_prices))
        )
        self.intraday_adjuster = (
            Adjuster(self.etf_prices_intraday)
            .add(FxSpotComponent(fx_composition, self.fx_prices_intraday))
            .add(DividendComponent(dividends, self.etf_prices_intraday,
                                   fx_prices=self.fx_prices_intraday))
        )

        self.corrected_return = pd.DataFrame(
            index=self.etf_prices.index, columns=self.etf_prices.columns, dtype=float)
        self.corrected_return_intraday = pd.DataFrame(
            index=self.etf_prices_intraday.index, columns=self.etf_prices_intraday.columns,
            dtype=float)

        # Modelli di pricing
        self.theoretical_live_cluster_price: Optional[pd.Series] = None
        self.theoretical_live_index_cluster_price: Optional[pd.Series] = None
        self.theoretical_intraday_prices: Optional[pd.Series] = None

        beta_cluster = (self.input_params.beta_cluster
                        .rename(self.ticker_to_isin, axis=1)
                        .rename(self.ticker_to_isin, axis=0))
        beta_cluster_index = (self.input_params.beta_cluster_index
                              .rename(self.ticker_to_isin, axis=1)
                              .rename(self.ticker_to_isin, axis=0))

        self.input_params.set_forecast_aggregation_func(kwargs["pricing"])

        self.cluster_model = ClusterPricingModel(
            beta=beta_cluster,
            returns=self.corrected_return,
            forecast_aggregator=self.input_params.forecast_aggregator_cluster,
            cluster_correction=self._calculate_cluster_correction(beta_cluster, 0),
            name="theoretical_live_cluster_price")

        self.cluster_model_intraday = ClusterPricingModel(
            beta=beta_cluster,
            returns=self.corrected_return_intraday,
            forecast_aggregator=TrimmedMean(0.2),
            cluster_correction=self._calculate_cluster_correction(beta_cluster_index, 0),
            name="theoretical_live_cluster_price")

        self.index_cluster_model = ClusterPricingModel(
            beta=beta_cluster_index,
            returns=self.corrected_return,
            forecast_aggregator=self.input_params.forecast_aggregator_cluster,
            cluster_correction=self._calculate_cluster_correction(beta_cluster_index, 0),
            name="theoretical_live_index_cluster_price")

        self.cluster_model.calculate_cluster_correction()
        self.index_cluster_model.calculate_cluster_correction()

    def _init_bloomberg(self, kwargs: dict) -> None:
        """Configura la sottoscrizione Bloomberg e pubblica i ritorni storici iniziali."""
        self.all_etf_plus_securities = list(set(self.currencies + self._isins_etf_equity))
        self.bloomberg_subscription_config_path = kwargs.get("bloomberg_subscription_config_path")
        self.subscription_manager = SubscriptionManager(
            self.all_etf_plus_securities, self.bloomberg_subscription_config_path)

        logger.info("=" * 70)
        logger.info("Price analytics: http://localhost:3000/d/etf-price-monitor-v2/")
        logger.info("=" * 70)

        static_return = self.adjuster.get_clean_returns()
        for i in self.return_to_publish:
            self.gui_redis.export_static_data(**{
                f"market:return_{i}": (static_return.iloc[-i].astype(float) * 100).round(4)
            })

    # =========================================================================
    # Market data setup
    # =========================================================================

    def on_market_data_setting(self) -> None:
        """Imposta i securities e le sottoscrizioni Bloomberg."""
        self.mid_eur = pd.Series(index=self.all_etf_plus_securities)
        self.book_eur = pd.DataFrame(columns=["BID", "ASK"])
        self.market_data.set_securities(self.all_etf_plus_securities)

        bloomberg_subscriptions = self.subscription_manager.get_subscription_dict()
        currency_info = self.subscription_manager.get_currency_informations()
        self.market_data.currency_information = currency_info

        subscription_manager = self.market_data.get_subscription_manager()

        for isin in self.etfs:
            subscription_manager.subscribe_bloomberg(
                id=isin,
                subscription_string=bloomberg_subscriptions.get(isin, isin),
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

        for ccy in self.currencies:
            subscription_manager.subscribe_bloomberg(
                id=ccy,
                subscription_string=f"{ccy} CURNCY",
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def wait_for_book_initialization(self):
        """Attende l'inizializzazione del book e gestisce strumenti con dati mancanti."""
        while datetime.today().time() < time(9, 5):
            return False

        while not self.wait_for_bloomberg_initialization():
            sleep_time.sleep(1)

        return True

    def wait_for_bloomberg_initialization(self):
        while self.market_data.get_pending_subscriptions("bloomberg"):
            sleep_time.sleep(1)

        subscription_manager = self.market_data.get_subscription_manager()
        for sub in subscription_manager.get_failed_subscriptions():
            isin = sub.get("id")
            if isin in self.etfs:
                ticker = self.isin_to_ticker.get(isin)
                subscription_manager.subscribe_bloomberg(isin, f"{ticker} IM EQUITY", ["BID", "ASK"])

        sleep_time.sleep(5)

        for sub in subscription_manager.get_failed_subscriptions():
            isin = sub.get("id")
            logger.warning(f"failed subscription: '{isin} IM EQUITY' ({self.isin_to_ticker.get(isin)})")
            self.failed_isin.append(isin)

        return True

    # =========================================================================
    # Loop ad alta frequenza
    # =========================================================================

    def update_HF(self):
        """Aggiornamento ad alta frequenza: prezzi, modelli, export GUI e storage TS."""
        if datetime.today().time() < time(17, 30):
            self.get_mid()
            self.calculate_cluster_theoretical_price()

            normalized_prices = {
                'live_idx':  self.round_series_to_tick(
                    self.theoretical_live_index_cluster_price, self.reference_tick_size),
                'live_clust': self.round_series_to_tick(
                    self.theoretical_live_cluster_price.fillna(0), self.reference_tick_size),
                'intraday':  self.round_series_to_tick(
                    self.theoretical_intraday_prices.fillna(0), self.reference_tick_size),
                'mid':       self.round_series_to_tick(
                    self.mid_eur.fillna(0), self.reference_tick_size),
            }

            self._export_normalized_prices_to_gui(normalized_prices)

            current_time = datetime.now()
            if ((current_time - self.last_storage_time).total_seconds() > 2
                    and self.timeseries_publisher is not None):
                self._publish_to_storage(normalized_prices, current_time)

    def update_LF(self):
        """Aggiornamento a bassa frequenza: pubblica i ritorni intraday storici."""
        fx = self.mid_eur[self.currencies]
        etfs = self.mid_eur[self.all_etf_plus_securities]
        self.intraday_adjuster.append_update(prices=etfs, fx_prices=fx)
        intraday_returns = self.intraday_adjuster.get_clean_returns()

        intraday_returns.index = intraday_returns.index.floor('min')
        intraday_returns.sort_index(ascending=False, inplace=True)
        intraday_returns.index = intraday_returns.index.strftime('%Y-%m-%dT%H:%M:%S')

        self.gui_redis.export_static_data(df_big=intraday_returns.T.to_json())

    # =========================================================================
    # Export prezzi alla GUI
    # =========================================================================

    def _export_normalized_prices_to_gui(self, normalized_prices: Dict[str, any]):
        """Esporta prezzi normalizzati via pub/sub (Redis o RabbitMQ)."""
        export_mapping = {
            'market:theoretical_live_index_cluster_price': 'live_idx',
            'market:theoretical_live_cluster_price':       'live_clust',
            'market:theoretical_live_intraday_price':      'intraday',
            'market:mid':                                  'mid',
        }
        for channel, price_key in export_mapping.items():
            try:
                self.gui_redis.export_message(
                    channel,
                    normalized_prices[price_key],
                    skip_if_unchanged=True,
                    flat_mode=True,
                )
            except Exception as e:
                logger.error(f"export_message failed for {channel}: {e}", exc_info=True)

    # =========================================================================
    # Pubblicazione Redis TimeSeries
    # =========================================================================

    def _build_ts_labels(self, isin: str, field: str) -> Dict[str, str]:
        """Costruisce le label TS: ISIN, TICKER, type (MID|MISALIGNMENT), model."""
        labels: Dict[str, str] = {
            'isin':   isin,
            'ticker': self.isin_to_ticker.get(isin, isin),
        }
        labels.update(self._TS_FIELD_META.get(field, {}))
        return labels

    def _publish_to_storage(self, normalized_prices: Dict[str, any], current_time: datetime):
        """
        Pubblica su Redis TS in un unico batch:
        - mid mercato (type=MID)
        - prezzi teorici live_idx, live_clust, intraday (type=MODEL_PRICE, model=...)
        - misalignment live_idx_mis, live_clust_mis, intraday_mis (type=MISALIGNMENT, model=...)
        """
        all_isins: Set[str] = set()
        for series in [self.theoretical_live_index_cluster_price,
                       self.theoretical_live_cluster_price,
                       self.theoretical_intraday_prices]:
            all_isins.update(series.keys())

        if not all_isins:
            logger.warning("No ISINs found for TS storage")
            return

        mid_prices = normalized_prices['mid']
        price_fields = [
            ('live_idx',   normalized_prices['live_idx']),
            ('live_clust', normalized_prices['live_clust']),
            ('intraday',   normalized_prices['intraday']),
        ]
        mis_fields = [
            ('live_idx_mis',   normalized_prices['live_idx']),
            ('live_clust_mis', normalized_prices['live_clust']),
            ('intraday_mis',   normalized_prices['intraday']),
        ]

        count = 0
        try:
            with self.timeseries_publisher.ts_batch() as batch:
                for isin in all_isins:
                    mid_val = mid_prices.get(isin)
                    if not mid_val or mid_val == 0 or np.isnan(mid_val):
                        continue

                    # MID mercato
                    batch.add(isin, 'mid', float(mid_val),
                              labels=self._build_ts_labels(isin, 'mid'))
                    count += 1

                    # Prezzi teorici (MID space)
                    for field, series in price_fields:
                        val = series.get(isin)
                        if val is not None and not np.isnan(val):
                            batch.add(isin, field, float(val),
                                      labels=self._build_ts_labels(isin, field))
                            count += 1

                    # Misalignment (scarto % dal mid)
                    for field, series in mis_fields:
                        val = series.get(isin)
                        if val is not None and not np.isnan(val):
                            mis = float(np.round(val / mid_val - 1, 6))
                            batch.add(isin, field, mis,
                                      labels=self._build_ts_labels(isin, field))
                            count += 1

        except Exception as e:
            logger.error(f"TS batch publishing failed: {e}", exc_info=True)
            raise

        self.last_storage_time = current_time
        logger.debug(f"TS storage published {count} values for {len(all_isins)} ISINs")

    # =========================================================================
    # Pricing
    # =========================================================================

    def get_mid(self) -> pd.Series:
        """
        Legge il mid corrente dal book, filtra outlier e aggiorna i ritorni corretti.
        """
        last_mid = self.market_data.get_mid()
        self.book_eur = self.market_data.get_data_field(["BID", "ASK"])

        if self.mid_eur is not None:
            safe_last_book = last_mid.replace(0, np.nan)
            is_outlier = (
                last_mid.isna()
                | (last_mid == 0)
                | ((self.mid_eur / safe_last_book - 1).abs() > self.book_mid_threshold)
            )
            valid_entries = last_mid[~is_outlier]
            self.mid_eur.loc[[i for i in valid_entries.index if i in self.mid_eur.index]] = valid_entries
        else:
            self.mid_eur = last_mid

        with self.adjuster.live_update(fx_prices=last_mid[self.currencies], prices=last_mid):
            self.corrected_return = self.adjuster.get_clean_returns(cumulative=True).T
            last_return = self.corrected_return.iloc[:, -1]

        with self.intraday_adjuster.live_update(fx_prices=last_mid[self.currencies], prices=last_mid):
            self.corrected_return_intraday = self.intraday_adjuster.get_clean_returns(cumulative=True).T
            last_return_intraday = self.corrected_return_intraday.iloc[:, -1]

        self.gui_redis.export_message("market:return_0",
                                      (last_return.astype(float) * 100).round(4))
        self.gui_redis.export_message("market:intraday_return_0",
                                      (last_return_intraday.astype(float) * 100).round(4))
        self.book_storage.append(last_mid)
        return last_mid

    def calculate_cluster_theoretical_price(self):
        """Calcola i prezzi teorici dai tre modelli cluster."""
        try:
            self.theoretical_live_cluster_price = self.cluster_model.get_price_prediction(
                self.mid_eur, self.corrected_return.T)
            self.theoretical_live_index_cluster_price = self.index_cluster_model.get_price_prediction(
                self.mid_eur, self.corrected_return.T)
            self.theoretical_intraday_prices = self.cluster_model_intraday.get_price_prediction(
                self.mid_eur, self.corrected_return_intraday.T)
        except Exception as e:
            logger.error(f"Error calculating cluster price: {e}")

    # =========================================================================
    # Utilities
    # =========================================================================

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = [self.as_isin(_id) for _id in set(value) if self.as_isin(_id)]

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments

    def as_isin(self, _id: str | list[str]) -> list[str] | str:
        if isinstance(_id, str):
            return self.ticker_to_isin.get(_id, _id)
        return [self.ticker_to_isin.get(el, el) for el in _id]

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        """Arrotonda una Series ai tick specificati per ciascun strumento."""
        if series is None:
            return series
        if isinstance(tick_dict, pd.Series):
            tick_dict = tick_dict.to_dict()
        ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index]) / 2
        values = series.fillna(0).values.astype(float)
        rounded_values = np.round(np.round(values / ticks) * ticks, 10)
        return pd.Series(rounded_values, index=series.index).fillna(0)

    @staticmethod
    def filter_outliers(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)
        if logger.isEnabledFor(logging.DEBUG) and outliers.any().any():
            out_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_parquet(os.path.join(out_dir, f"{name}_raw_{ts}.parquet"))
            outliers.to_parquet(os.path.join(out_dir, f"{name}_outliers_{ts}.parquet"))
            logger.debug(f"Outliers in {name}: {outliers.sum().sum()} values, saved to {out_dir}")
        df[outliers] = np.nan
        return df

    @staticmethod
    def _calculate_cluster_correction(cluster_betas: pd.DataFrame,
                                      threshold: float = 0.5) -> pd.Series:
        """
        Calcola il fattore di correzione per ciascun sottocluster.

        Returns:
            pd.Series: fattori di correzione per ISIN.
        """
        cluster_betas = cluster_betas.sort_index(axis=1).sort_index(axis=0)
        for etf in cluster_betas.index:
            cluster_betas.loc[etf, etf] = 0
        cluster_threshold: pd.Series = threshold / (cluster_betas != 0).sum(axis=1)
        cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1) + 1
        return cluster_sizes.where(cluster_sizes == 1, (cluster_sizes - 1) / cluster_sizes)

    def get_update_hf_stats(self) -> dict:
        """Ritorna statistiche dell'ultimo ciclo update_HF."""
        ts_stats = self.timeseries_publisher.ts_stats if self.timeseries_publisher else {}
        return {
            "timeseries_created": ts_stats.get("timeseries_created", 0),
            "total_published":    ts_stats.get("total_published", 0),
            "duplicates_skipped": ts_stats.get("duplicates_skipped", 0),
            "errors":             ts_stats.get("errors", 0),
            "last_storage_time":  self.last_storage_time.isoformat()
                                  if hasattr(self, 'last_storage_time') else None,
        }
