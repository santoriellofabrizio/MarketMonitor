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
from core.enums.currencies import CurrencyEnum
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


class EtfEquityPriceEngine(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\BshDataProvider\config\bshdata_config.yaml")

        isins_etf_equity = self.API.general.get_etp_isins(underlying="EQUITY", segments="IM")
        self.reference_tick_size = self.API.info.get_etp_fields(isin=isins_etf_equity,
                                                                fields=["REFERENCE_TICK_SIZE"],
                                                                source="bloomberg")["REFERENCE_TICK_SIZE"].to_dict()

        self.isin_to_ticker = self.API.info.get_etp_fields(isin=isins_etf_equity, source="bloomberg",
                                                           fields=["TICKER"])["TICKER"].to_dict()

        self.currencies = [f"EUR{c}" for c in CurrencyEnum.__members__ if c != "EUR"]
        self.gui_redis = RedisMessaging()
        self.gui_redis.export_static_data(ENGINE_PID=os.getpid())

        self.mid_eur: Optional[pd.Series] = None
        self.book_mid_threshold = .5
        self.subscription_summary: pd.DataFrame | None = None
        self.input_params = InputParamsQuoting(**kwargs)
        self.subscription_manager: None | SubscriptionManager = None
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        self.number_of_days = kwargs.get("number_of_days", 5)

        self.theoretical_prices: pd.DataFrame | None = None
        self.book_storage: deque = deque(maxlen=3)
        self.spreads: Optional[pd.Series] = None
        self.strategy_input: pd.DataFrame | pd.Series | None = None
        self.position: Optional[pd.Series] = None
        self.live_weights: Optional[pd.Series] = None
        self.threshold_exceeded_instruments = set()
        self.return_to_publish: list = [1, 2, 3, 4, 5, 6, 7, 8]
        self._cumulative_returns: bool = True

        self.fx_list: list | None = None
        self.mid_eur: pd.Series()
        self.book_eur: pd.DataFrame()
        self.securities_list: list | None = None
        self.instruments_status: None | pd.Series = None
        self.GUIs: GUI
        self.ask_for_stock_drop: bool = kwargs.get("ask_for_stock_drop", False)
        self.today = pd.Timestamp.today().normalize()
        self.yesterday = HolidayManager().previous_business_day(self.today)

        self.instruments_cluster = self.input_params.isin_cluster

        cluster_price_regressor = list(set(isins_etf_equity).
                                       intersection(self.as_isin(self.input_params.beta_cluster.columns)))
        self.instruments = list(set(self.instruments_cluster + cluster_price_regressor))

        self.holidays = HolidayManager()
        start = self.holidays.subtract_business_days(today(), self.number_of_days)
        end = yesterday = self.holidays.previous_business_day(today())
        days = self.holidays.get_business_days(start=start, end=end)
        snapshot_time = time(16)

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

        self.fx_prices_intraday = self.API.market.get_intraday_fx(id=self.currencies,
                                                                  date=yesterday,
                                                                  start_time=time(10),
                                                                  frequency="15m")

        self.etf_prices_intraday = self.API.market.get_intraday_etf(id=self.instruments,
                                                                    date=yesterday,
                                                                    start_time=time(10),
                                                                    frequency="15m")

        # _, fx_full = self.input_params.get_currency_data(self.instruments)

        fx_forward_needed = fx_forward.columns.tolist()

        fx_forward_prices = self.API.market.get_daily_fx_forward(quoted_currency=fx_forward_needed,
                                                                 start=start,
                                                                 end=end)

        dividends = self.API.info.get_dividends(id=self.instruments)
        ter = self.API.info.get_ter(id=self.instruments)/100
        ter.update(self.input_params.ter_manual)

        self.adjuster = (
            Adjuster(self.etf_prices, intraday=False)
            .add(TerComponent(ter))
            .add(FxSpotComponent(fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(fx_forward, fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(dividends, fx_prices=self.fx_prices))
        )

        self.intraday_adjuster = (
            Adjuster(self.etf_prices_intraday)
            .add(FxSpotComponent(fx_composition, self.fx_prices_intraday))
            .add(DividendComponent(dividends, fx_prices=self.fx_prices))
        )

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
            beta=beta_cluster_index,
            returns=self.corrected_return,
            forecast_aggregator=self.input_params.forecast_aggregator_cluster,
            name="theoretical_live_cluster_price")

        self.cluster_model_intraday = ClusterPricingModel(
            beta=beta_cluster,
            returns=self.corrected_return_intraday,
            forecast_aggregator=TrimmedMean(0.2),
            name="theoretical_live_cluster_price")

        self.index_cluster_model = ClusterPricingModel(
            beta=beta_cluster_index,
            returns=self.corrected_return,
            forecast_aggregator=self.input_params.forecast_aggregator_cluster,
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
        # self.check_book_update(1200)
        pass

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
            for isin in last_mid[is_outlier].index:
                self.threshold_exceeded_instruments.add(isin)
        else:
            self.mid_eur = last_mid

        self.corrected_return = self.adjuster.clean_returns(cumulative=True,
                                                            fx_prices=last_mid[self.currencies],
                                                            live_prices=last_mid).T

        self.corrected_return_intraday = self.intraday_adjuster.clean_returns(cumulative=True,
                                                                              fx_prices=last_mid[self.currencies],
                                                                              live_prices=last_mid).T

        # Publish returns
        for i in self.return_to_publish:
            self.gui_redis.export_message(f"market:{i}D_return",
                                          self.corrected_return.iloc[:, -i].astype(float).round(4))

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
                                                                     self.corrected_return.T))
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
