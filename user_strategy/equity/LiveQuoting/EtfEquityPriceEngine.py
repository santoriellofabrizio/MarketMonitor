import logging
import os
import sqlite3
import time as sleep_time

from collections import deque
from datetime import datetime, time, date
from typing import Optional, Set
import numpy as np
import pandas as pd
from dateutil.utils import today

from sfm_data_provider.analytics.adjustments.adjuster import Adjuster
from sfm_data_provider.analytics.adjustments.dividend import DividendComponent
from sfm_data_provider.analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from sfm_data_provider.analytics.adjustments.fx_spot import FxSpotComponent
from sfm_data_provider.analytics.adjustments.ter import TerComponent
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager

from market_monitor.publishers.timeseries_publisher import TimeSeriesPublisher
from user_strategy.equity.utils.SQLUtils.storage import PriceDatabaseManager
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.gui.implementations.GUI import GUI
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.equity.LiveQuoting.InputParamsQuoting import InputParamsQuoting

from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator, TrimmedMean

from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager

logger = logging.getLogger(__name__)

DB_ANAGRPHIC_PATH = r"V:\EquityETF\etf_equity_anagraphic_db.sqlite"


class EtfEquityPriceEngine(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.number_of_days_intraday = kwargs.get("number_of_days_intraday", 3)
        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\MarketMonitor\etc\config\bshdata_config.yaml")

        self.failed_isin = []

        isins_etf_equity = self.API.general.get(fields=["etp_isins"],
                                                segments=["IM"],
                                                currency="EUR",
                                                underlying="EQUITY",
                                                source="oracle")["etp_isins"]

        self.reference_tick_size = self.API.info.get_etp_fields(isin=isins_etf_equity,
                                                                fields=["REFERENCE_TICK_SIZE"],
                                                                source="bloomberg")["REFERENCE_TICK_SIZE"].to_dict()
        try:
            self.timeseries_publisher = TimeSeriesPublisher()
        except Exception as e:
            self.timeseries_publisher = None
            self.logger.warning("redis TS not connected,", e)

        with sqlite3.connect(DB_ANAGRPHIC_PATH) as conn:
            self.isin_to_ticker = pd.read_sql(
                "SELECT ISIN, TICKER FROM vw_isin_ticker", conn
            ).set_index("ISIN")["TICKER"].to_dict()

            self.active_isin = pd.read_sql(
                "SELECT ISIN, MARKET_STATUS FROM market_status", conn
            ).set_index("ISIN")["MARKET_STATUS"].to_dict()

        currencies = ['USD', 'GBP', 'CHF', 'AUD', 'DKK', 'HKD', 'NOK',
                      'PLN', 'SEK', 'CNY', 'JPY', 'CNH', 'CAD', 'INR', 'BRL']

        self.currencies = [f"EUR{c}" for c in currencies if c != "EUR"]
        self.gui_redis = RedisMessaging()

        self.mid_eur: Optional[pd.Series] = None
        self.book_mid_threshold = .5
        self.input_params = InputParamsQuoting(**kwargs)
        self.subscription_manager: None | SubscriptionManager = None
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        self.number_of_days = kwargs.get("number_of_days", 5)

        self.theoretical_prices: pd.DataFrame | None = None
        self.book_storage: deque = deque(maxlen=3)
        self.strategy_input: pd.DataFrame | pd.Series | None = None
        self.position: Optional[pd.Series] = None
        self.return_to_publish: list = [1, 2, 3, 4, 5, 6, 7, 8]
        self._cumulative_returns: bool = True

        self.fx_list: list | None = None
        self.mid_eur: pd.Series()
        self.book_eur: pd.DataFrame()
        self.securities_list: list | None = None
        self.instruments_status: None | pd.Series = None
        self.GUIs: GUI
        self.today = pd.Timestamp.today().normalize()
        self.yesterday = HolidayManager().previous_business_day(self.today)

        beta_cluster = self.input_params.beta_cluster
        beta_cluster_index = self.input_params.beta_cluster_index

        self.etfs = list({
            *self.input_params.beta_cluster.index,
            *self.input_params.beta_cluster.columns,
            *self.input_params.beta_cluster_index.index,
            *self.input_params.beta_cluster_index.columns,
        })

        for etf in self.etfs:
            if self.active_isin.get(etf, "NOT_PRESENT") != "ACTV":
                logger.warning(f"{etf} ({self.ticker_to_isin.get(etf)}) is not ACTV.")
                self.etfs.remove(etf)

        for attr in ['beta_cluster', 'beta_cluster_index']:
            df = getattr(self.input_params, attr)
            # Filtra solo per le colonne/indici rimasti
            df_filtered = df.loc[df.index.intersection(self.etfs), df.columns.intersection(self.etfs)]
            # Rinormalizza dividendo per la nuova somma
            setattr(self.input_params, attr, df_filtered.div(df_filtered.sum(axis=1), axis=0).fillna(0))

        self.instruments = list({
            *self.etfs,
            *self.currencies
        })

        self.holidays = HolidayManager()
        start = self.holidays.subtract_business_days(today(), self.number_of_days)
        start_intraday = self.holidays.subtract_business_days(today(), self.number_of_days_intraday)
        end = self.holidays.previous_business_day(today())
        days = self.holidays.get_business_days(start=start, end=end)
        snapshot_time = time(16, 45)
        self.last_storage_time = datetime.now()

        fx_composition = self.API.info.get_fx_composition(self.etfs, fx_fxfwrd="fx",
                                                          reference_date=date(2025, 12, 18))

        fx_forward = self.API.info.get_fx_composition(self.etfs, fx_fxfwrd="fxfwrd",
                                                      reference_date=date(2025, 12, 18))

        for isin, isin_proxy in kwargs.get("fx_mapping", {}).items():
            fx_composition.loc[isin] = fx_composition.loc[isin_proxy]

        self.fx_prices = self.API.market.get_daily_currency(id=self.currencies,
                                                            start=start,
                                                            end=end,
                                                            snapshot_time=snapshot_time,
                                                            fallbacks=[{"source": "bloomberg"}]).reindex(days)

        self.etf_prices = self.API.market.get_daily_etf(id=self.etfs,
                                                        start=start,
                                                        end=end,
                                                        snapshot_time=snapshot_time,
                                                        fallbacks=[{"source": "bloomberg",
                                                                    "market": "IM"}]).reindex(days)

        self.fx_prices_intraday = self.filter_outliers(
            self.API.market.get_intraday_fx(id=self.currencies,
                                            start=start_intraday,
                                            end=end,
                                            frequency="15m",
                                            fallbacks=[{"source": "bloomberg",
                                                        "market": "IM"}])
            .between_time("10:00", "17:00"))

        self.etf_prices_intraday = self.filter_outliers(
            self.API.market.get_intraday_etf(id=self.etfs,
                                             start=start_intraday,
                                             end=end,
                                             frequency="15m",
                                             source='bloomberg',
                                             fallbacks=[{"source": "bloomberg", "market": "IM"}])
            .between_time("10:00", "17:00"), name="etf_intraday")

        self.etf_prices = self.etf_prices.interpolate("time")
        self.etf_prices_intraday = self.etf_prices_intraday.interpolate("time")

        # _, fx_full = self.input_params.get_currency_data(self.instruments)

        fx_forward_needed = fx_forward.columns.tolist()

        fx_forward_prices = self.API.market.get_daily_fx_forward(quoted_currency=fx_forward_needed,
                                                                 start=start,
                                                                 end=end)

        dividends = self.API.info.get_dividends(id=self.etfs, start=start)
        ter = self.API.info.get_ter(id=self.etfs) / 100

        self.adjuster = (
            Adjuster(self.etf_prices)
            .add(TerComponent(ter))
            .add(FxSpotComponent(fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(fx_forward, fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(dividends, self.etf_prices, fx_prices=self.fx_prices))
        )

        self.intraday_adjuster = Adjuster(self.etf_prices_intraday).add(
            FxSpotComponent(fx_composition, self.fx_prices_intraday)).add(
            DividendComponent(dividends, self.etf_prices_intraday, fx_prices=self.fx_prices_intraday))

        self.corrected_return: Optional[pd.DataFrame] = pd.DataFrame(index=self.etf_prices.index,
                                                                     columns=self.etf_prices.columns,
                                                                     dtype=float)

        self.corrected_return_intraday: Optional[pd.DataFrame] = pd.DataFrame(index=self.etf_prices_intraday.index,
                                                                              columns=self.etf_prices_intraday.columns,
                                                                              dtype=float)

        self.bloomberg_subscription_config_path = kwargs.get("bloomberg_subscription_config_path", None)
        self.all_etf_plus_securities = list(set(self.currencies + list(isins_etf_equity)))
        self.subscription_manager = SubscriptionManager(self.all_etf_plus_securities,
                                                        self.bloomberg_subscription_config_path)

        # ----------------------------------------- PRICING ------------------------------------------------------------

        self.theoretical_live_cluster_price: Optional[pd.Series] = None
        self.theoretical_live_index_cluster_price: pd.Series | None = None
        self.theoretical_intraday_prices: pd.Series | None = None

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
            beta=beta_cluster_index,
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

        # Publish returns
        static_return = self.adjuster.get_clean_returns()
         for i in self.return_to_publish:
            self.gui_redis.export_static_data(**{f"market:return_{i}":
                                                     (static_return.iloc[-i].astype(float) * 100).round(4)})

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        default for ccy is EURCCY. es EURUSD.
        """
        self.mid_eur = pd.Series(index=self.all_etf_plus_securities)
        self.book_eur = pd.DataFrame(columns=["BID", "ASK"])
        self.market_data.set_securities(self.all_etf_plus_securities)

        # Get subscription info
        bloomberg_subscriptions = self.subscription_manager.get_subscription_dict()
        currency_info = self.subscription_manager.get_currency_informations()

        # Set currency information
        self.market_data.currency_information = currency_info

        subscription_manager = self.market_data.get_subscription_manager()

        # Subscribe using new Bloomberg API
        for isin in self.etfs:
            # Determine if currency

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
        """
        Attende l'inizializzazione del book e gestisce strumenti con dati mancanti.
        """

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
            logger.warning(f"failed subscription: -> '{isin} IM EQUITY' ({self.isin_to_ticker.get(isin)})")
            self.failed_isin.append(isin)

        return True

    """
    Versione ottimizzata della funzione update_HF con:
    - Creazione TimeSeries centralizzata e una sola volta
    - Calcoli di misalignment pre-elaborati
    - Pipeline Redis per export message
    - Batch publishing unificato
    - Eliminazione ridondanze
    """

    from datetime import datetime, time
    import numpy as np
    from typing import Dict, Tuple, Set
    import logging

    logger = logging.getLogger(__name__)

    """
    Versione ottimizzata della funzione update_HF con:
    - Creazione TimeSeries centralizzata e una sola volta
    - Calcoli di misalignment pre-elaborati
    - Uso corretto di RedisMessaging per export GUI
    - Batch publishing unificato
    - Eliminazione ridondanze

    IMPORTANTE: Usa self.gui_redis.export_message() NON redis_client.pipeline()
    per mantenere correttezza di:
      - Normalizzazione dati
      - Change detection
      - Serializzazione JSON
      - Size control (Excel RTD limit)
      - Logging tracciato
    """

    from datetime import datetime, time
    import numpy as np
    from typing import Dict, Tuple, Set, Optional, Any
    import logging

    logger = logging.getLogger(__name__)

    def update_HF(self):
        """Aggiornamento ad alta frequenza con ottimizzazioni di performance."""

        # 1. UPDATE DATI
        if datetime.today().time() < time(17, 30):
            self.get_mid()

            self.calculate_cluster_theoretical_price()

            # 2. PREPARAZIONE SERIE NORMALIZZATE (una sola volta)
            normalized_prices = {
                'live_idx': self.round_series_to_tick(
                    self.theoretical_live_index_cluster_price,
                    self.reference_tick_size
                ),
                'live_clust': self.round_series_to_tick(
                    self.theoretical_live_cluster_price.fillna(0),
                    self.reference_tick_size
                ),
                'intraday': self.round_series_to_tick(
                    self.theoretical_intraday_prices.fillna(0),
                    self.reference_tick_size
                ),
                'mid': self.round_series_to_tick(
                    self.mid_eur.fillna(0),
                    self.reference_tick_size
                )
            }

            # 3. EXPORT ALLE GUI (pipeline Redis)
            self._export_normalized_prices_to_gui(normalized_prices)

            # 4. STORAGE - Solo se è passato il tempo minimo
            current_time = datetime.now()
            if (current_time - self.last_storage_time).total_seconds() > 2 and self.timeseries_publisher is not None:
                self._publish_to_storage(normalized_prices, current_time)

    def _export_normalized_prices_to_gui(self, normalized_prices: Dict[str, any]):
        """
        Esporta prezzi normalizzati alle GUI usando RedisMessaging.

        Mantiene normalizzazione, change detection, e serializzazione corretti.
        Ottimizzazione: batch tutte le operazioni insieme anziché una per una.
        """
        export_mapping = {
            'market:theoretical_live_index_cluster_price': 'live_idx',
            'market:theoretical_live_cluster_price': 'live_clust',
            'market:theoretical_live_intraday_price': 'intraday',
            'market:mid': 'mid'
        }

        # Pubblica attraverso RedisMessaging per mantenere:
        # - Normalizzazione (già fatto, ma passa through)
        # - Change detection
        # - Serializzazione con metadata
        # - Logging tracciato
        for channel, price_key in export_mapping.items():
            try:
                self.gui_redis.export_message(
                    channel,
                    normalized_prices[price_key],
                    skip_if_unchanged=True,  # ✅ Salta se non è cambiato
                    flat_mode=True  # ✅ RTD-compatible format
                )
            except Exception as e:
                logger.error(f"Errore export_message per {channel}: {e}", exc_info=True)

        logger.debug(f"GUI export complete: {len(export_mapping)} channels processed")

    def _publish_to_storage(self, normalized_prices: Dict[str, any], current_time: datetime):
        """
        Pubblica dati su Redis TimeSeries con batch ottimizzato.

        Ottimizzazioni:
        - Crea TimeSeries una sola volta per ISINs trovati
        - Calcola misalignment pre-batch
        - Usa ts_batch() per pubblicare tutto insieme
        """

        # 1. RACCOLTA ISINS - Una sola volta
        all_isins: Set[str] = set()
        price_series = [
            self.theoretical_live_index_cluster_price,
            self.theoretical_live_cluster_price,
            self.theoretical_intraday_prices,
        ]

        for series in price_series:
            all_isins.update(series.keys())

        if not all_isins:
            logger.warning("No ISINs found for storage")
            return

        # 2. CREAZIONE TIMESERIES - Una sola volta per ISIN
        field_names = ['live_idx_mis', 'live_clust_mis', 'intraday_mis']
        self._ensure_timeseries_exist(all_isins, field_names)

        # 3. PREPARAZIONE MISALIGNMENTS - Pre-elaborati
        misalignments = self._calculate_misalignments(
            normalized_prices,
            all_isins
        )

        # 4. BATCH PUBLISHING - Una sola volta
        self._batch_publish_misalignments(misalignments)

        self.last_storage_time = current_time
        logger.info(f"Storage published {len(misalignments)} entries")

    def _ensure_timeseries_exist(self, isins: Set[str], field_names: list):
        """
        Crea TimeSeries per tutte le combinazioni ISIN x field_name.

        Ottimizzazione: usa cache interno di TimeSeriesPublisher.
        """
        created_count = 0
        initial_count = self.timeseries_publisher.ts_stats.get("timeseries_created", 0)

        for isin in isins:
            for field_name in field_names:
                try:
                    self.timeseries_publisher.ts_create(isin, field_name)
                    created_count += 1
                except Exception as e:
                    # Già esiste o errore recoverable
                    logger.debug(f"ts_create for {isin}:{field_name}: {e}")

        new_created = (self.timeseries_publisher.ts_stats.get("timeseries_created", 0)
                       - initial_count)
        if new_created > 0:
            logger.debug(f"Created {new_created} new TimeSeries")

    def _calculate_misalignments(
            self,
            normalized_prices: Dict[str, any],
            all_isins: Set[str]
    ) -> list:
        """
        Calcola misalignment (scarto %) per tutti gli ISINs.

        Ritorna lista di tuple (isin, field_name, misalignment_value)
        pronta per batch publish.

        Ottimizzazione: vectorizzato per ISINs, non per ogni ISIN dentro il loop.
        """
        misalignments = []

        mid_prices = normalized_prices['mid']

        price_mapping = [
            ('live_idx_mis', normalized_prices['live_idx']),
            ('live_clust_mis', normalized_prices['live_clust']),
            ('intraday_mis', normalized_prices['intraday'])
        ]

        for field_name, price_series in price_mapping:
            for isin in all_isins:
                try:
                    series_value = price_series.get(isin)
                    mid_price = mid_prices.get(isin)

                    # Validazione: mid_price deve essere positivo e non NaN
                    if mid_price is None or mid_price == 0 or np.isnan(mid_price):
                        continue

                    # Calcolo misalignment
                    if series_value is not None and not np.isnan(series_value):
                        misalignment = float(np.round(series_value / mid_price - 1, 6))
                        misalignments.append((isin, field_name, misalignment))

                except (TypeError, ZeroDivisionError, ValueError) as e:
                    logger.warning(f"Error calculating misalignment for {isin}: {e}")
                    continue

        return misalignments

    def _batch_publish_misalignments(self, misalignments: list):
        """
        Pubblica misalignments in un singolo batch.

        Ottimizzazione: usa ts_batch() context manager per minimo overhead.
        """
        if not misalignments:
            logger.warning("No misalignments to publish")
            return

        try:
            with self.timeseries_publisher.ts_batch() as batch:
                for isin, field_name, misalignment_value in misalignments:
                    batch.add(isin, field_name, misalignment_value)

            logger.debug(f"Batch published {len(misalignments)} misalignment values")

        except Exception as e:
            logger.error(f"Batch publishing failed: {e}", exc_info=True)
            raise

    # ============================================================================
    # Statistiche e Monitoring (opzionale)
    # ============================================================================

    def get_update_hf_stats(self) -> dict:
        """Ritorna statistiche dell'ultimo update_HF."""
        ts_stats = self.timeseries_publisher.ts_get_stats()
        return {
            "timeseries_created": ts_stats.get("timeseries_created", 0),
            "total_published": ts_stats.get("total_published", 0),
            "duplicates_skipped": ts_stats.get("duplicates_skipped", 0),
            "errors": ts_stats.get("errors", 0),
            "last_storage_time": self.last_storage_time.isoformat() if hasattr(self, 'last_storage_time') else None
        }

    def update_LF(self):

        fx = self.mid_eur[self.currencies]
        etfs = self.mid_eur[self.all_etf_plus_securities]
        self.intraday_adjuster.append_update(prices=etfs, fx_prices=fx)
        intraday_returns = self.intraday_adjuster.get_clean_returns()

        intraday_returns.index = intraday_returns.index.floor('min')
        intraday_returns.sort_index(ascending=False, inplace=True)
        intraday_returns.index = intraday_returns.index.strftime('%Y-%m-%dT%H:%M:%S')

        self.gui_redis.export_static_data(df_big=intraday_returns.T.to_json())

    def get_mid(self) -> pd.Series:
        """
        Get the mid-price of book.
        Store corrected returns and a copy of last book

        Returns:
            pd.Series: Series of md-prices for ETFs, Drivers, and FX.
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
        if isinstance(_id, str): return self.ticker_to_isin.get(_id, _id)
        return [self.ticker_to_isin.get(el, el) for el in _id]

    def calculate_cluster_theoretical_price(self):
        try:
            self.theoretical_live_cluster_price = (self.cluster_model.
                                                   get_price_prediction(self.mid_eur,
                                                                        self.corrected_return.T))
            self.theoretical_live_index_cluster_price = (self.index_cluster_model.
                                                         get_price_prediction(self.mid_eur,
                                                                              self.corrected_return.T))

            self.theoretical_intraday_prices = (self.cluster_model_intraday.
                                                get_price_prediction(self.mid_eur,
                                                                     self.corrected_return_intraday.T))
        except Exception as e:
            logging.error(f"Exception occurred while calculating cluster price: {e}")

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        """ Arrotonda una Series ai tick specificati per ciascun strumento e normalizza i float. """
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
    def _calculate_cluster_correction(cluster_betas: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Calculate the cluster correction factor for each subcluster.

        Returns:
            pd.Series: Series with correction factors for each ISIN.
        """
        # this first line is used for the brothers matrix, in order to make it comparable with the clusters matrix
        cluster_betas = cluster_betas.sort_index(axis=1)
        cluster_betas = cluster_betas.sort_index(axis=0)
        for label in cluster_betas.index:
            cluster_betas.loc[label, label] = 0
        # with the first series we define which is the threshold for a betas to be considered
        cluster_threshold: pd.Series = threshold / (cluster_betas != 0).sum(axis=1)
        # here we count only the beta which are above the threshold
        cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1) + 1
        # the correction is than calculated as the number of elements which truly influence our calculations
        correction = cluster_sizes.where(cluster_sizes == 1, (cluster_sizes - 1) / cluster_sizes)
        return correction
