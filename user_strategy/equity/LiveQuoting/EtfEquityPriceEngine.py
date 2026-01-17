import logging
import os
import time
from collections import deque
from datetime import datetime, time, date
from typing import Optional
import numpy as np
import pandas as pd
from dateutil.utils import today

from analytics.adjustments import Adjuster
from analytics.adjustments.dividend import DividendComponent
from analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from analytics.adjustments.fx_spot import FxSpotComponent
from analytics.adjustments.ter import TerComponent
from core.holidays.holiday_manager import HolidayManager
from interface.bshdata import BshData
from market_monitor.gui.implementations.GUI import GUI
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.equity.LiveQuoting.InputParamsQuoting import InputParamsQuoting, \
    DISMISSED_ETFS
from user_strategy.utils.pricing_models.AggregationFunctions import ForecastAggregator, TrimmedMean

from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager


logger = logging.getLogger(__name__)

class EtfEquityPriceEngine(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.number_of_days_intraday = kwargs.get("number_of_days_intraday", 3)
        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\BshDataProvider\config\bshdata_config.yaml")

        isins_etf_equity = self.API.general.get_etp_isins(underlying="EQUITY", segments="IM")
        self.reference_tick_size = self.API.info.get_etp_fields(isin=isins_etf_equity,
                                                                fields=["REFERENCE_TICK_SIZE"],
                                                                source="bloomberg")["REFERENCE_TICK_SIZE"].to_dict()

        self.isin_to_ticker = self.API.info.get_etp_fields(isin=isins_etf_equity, source="bloomberg",
                                                           fields=["TICKER"])["TICKER"].to_dict()

        currencies = ['USD', 'GBP', 'CHF', 'AUD', 'DKK', 'HKD', 'NOK', 'PLN', 'SEK', 'CNY', 'JPY', 'CNH', 'CAD', 'INR',
                      'BRL']

        self.currencies = [f"EUR{c}" for c in currencies if c != "EUR"]
        self.gui_redis = RedisMessaging()
        self.gui_redis.export_static_data(ENGINE_PID=os.getpid())

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

        self.instruments_cluster = self.input_params.isin_cluster

        cluster_price_regressor = list(set(isins_etf_equity).
                                       intersection(self.as_isin(self.input_params.beta_cluster.columns)))
        self.instruments = list(set(self.instruments_cluster + cluster_price_regressor))

        self.holidays = HolidayManager()
        start = self.holidays.subtract_business_days(today(), self.number_of_days)
        start_intraday = self.holidays.subtract_business_days(today(), self.number_of_days_intraday)
        end = yesterday = self.holidays.previous_business_day(today())
        days = self.holidays.get_business_days(start=start, end=end)
        snapshot_time = time(16, 45)

        fx_composition = self.API.info.get_fx_composition(self.instruments, fx_fxfwrd="fx",
                                                          reference_date=date(2025, 12, 18))

        fx_forward = self.API.info.get_fx_composition(self.instruments, fx_fxfwrd="fxfwrd",
                                                      reference_date=date(2025, 12, 18))

        self.fx_prices = self.API.market.get_daily_currency(id=self.currencies,
                                                            start=start,
                                                            end=end,
                                                            snapshot_time=snapshot_time,
                                                            fallbacks=[{"source": "bloomberg"}]).reindex(days)

        self.etf_prices = self.API.market.get_daily_etf(id=self.instruments,
                                                        start=start,
                                                        end=end,
                                                        snapshot_time=snapshot_time,
                                                        fallbacks=[{"source": "bloomberg", "market": "IM"}]).reindex(
            days)

        self.fx_prices_intraday = self.filter_outliers(
            self.API.market.get_intraday_fx(id=self.currencies,
                                            start=start_intraday,
                                            end=end,
                                            frequency="15m",
                                            fallbacks=[{"source":"bloomberg"}])
            .between_time("10:00", "17:00"))

        self.etf_prices_intraday = self.filter_outliers(
            self.API.market.get_intraday_etf(id=self.instruments,
                                             start=start_intraday,
                                             end=end,
                                             frequency="15m",
                                             fallbacks=[{"source":"bloomberg","market": "IM"}])
            .between_time("10:00", "17:00"), name="etf_intraday")

        self.etf_prices = self.etf_prices.interpolate("time")
        self.etf_prices_intraday = self.etf_prices_intraday.interpolate("time")

        # _, fx_full = self.input_params.get_currency_data(self.instruments)

        fx_forward_needed = fx_forward.columns.tolist()

        fx_forward_prices = self.API.market.get_daily_fx_forward(quoted_currency=fx_forward_needed,
                                                                 start=start,
                                                                 end=end)

        dividends = self.API.info.get_dividends(id=self.instruments, start=start)
        ter = self.API.info.get_ter(id=self.instruments) / 100
        ter.update(self.input_params.ter_manual)

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
        for id, subscription_string in bloomberg_subscriptions.items():
            # Determine if currency

            subscription_manager.subscribe_bloomberg(
                id=id,
                subscription_string=subscription_string,
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def wait_for_book_initialization(self):
        """
        Attende l'inizializzazione del book e gestisce strumenti con dati mancanti.
        """

        while datetime.today().time() < time(9, 5):
            return False

        return True

    def update_HF(self):

        if datetime.today().time() < time(17, 30):
            self.get_mid()
        self.calculate_cluster_theoretical_price()
        self.gui_redis.export_message("market:theoretical_live_index_cluster_price",
                                      self.round_series_to_tick(self.theoretical_live_index_cluster_price,
                                                                self.reference_tick_size))

        self.gui_redis.export_message("market:theoretical_live_cluster_price",
                                      self.round_series_to_tick(self.theoretical_live_cluster_price.fillna(0),
                                                                self.reference_tick_size))

        self.gui_redis.export_message("market:theoretical_live_intraday_price",
                                      self.round_series_to_tick(self.theoretical_intraday_prices.fillna(0),
                                                                self.reference_tick_size))

        self.gui_redis.export_message("market:mid",
                                      self.round_series_to_tick(self.mid_eur.fillna(0), self.reference_tick_size))

    def update_LF(self):
        fx = self.mid_eur[self.currencies]
        etfs = self.mid_eur[self.all_etf_plus_securities]
        self.intraday_adjuster.append_update(prices=etfs, fx_prices=fx)

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
        with self.intraday_adjuster.live_update(fx_prices=last_mid[self.currencies], prices=last_mid):
            self.corrected_return_intraday = self.intraday_adjuster.get_clean_returns(cumulative=True).T

        # Publish returns
        for i in self.return_to_publish:
            self.gui_redis.export_message(f"market:{i}D_return",
                                          self.corrected_return.iloc[:, i-1].astype(float).round(4))

        self.book_storage.append(last_mid)
        return last_mid

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = [self.as_isin(_id) for _id in set(value) if self.as_isin(_id) not in DISMISSED_ETFS]

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
