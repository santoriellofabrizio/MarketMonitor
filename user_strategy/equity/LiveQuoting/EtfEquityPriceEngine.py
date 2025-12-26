import logging
import os
import time
from collections import deque
from datetime import datetime, time
from typing import Optional
import numpy as np
import pandas as pd
from dateutil.utils import today
from sfm_return_adjustments_lib.ReturnAdjuster import ReturnAdjuster

from analytics.adjustments import Adjuster
from analytics.adjustments.dividend import DividendComponent
from analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from analytics.adjustments.fx_spot import FxSpotComponent
from analytics.adjustments.ter import TerComponent
from market_monitor.gui.implementations.GUI import GUI
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from core.holidays.holiday_manager import HolidayManager

from interface.bshdata import BshData
from user_strategy.equity.LiveQuoting.InputParamsQuoting import InputParamsQuoting, \
    DISMISSED_ETFS
from user_strategy.equity.utils.DataProcessors.PCFControls import PCFControls
from user_strategy.equity.utils.DataProcessors.PCFProcessor import PCFProcessor
from user_strategy.equity.utils.DataProcessors.StockSelector import StockSelector
from user_strategy.utils import CustomBDay

from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager
from user_strategy.utils.enums import ISIN_TO_TICKER, CURRENCY, ISINS_ETF_EQUITY, TICK_SIZE


class EtfEquityPriceEngine(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.gui_redis = RedisMessaging()
        self.gui_redis.export_static_data(ENGINE_PID=os.getpid())

        self.mid_eur: Optional[pd.Series] = None
        self.nav_bid_ask: Optional[pd.Series] = pd.Series()
        self.book_mid_threshold = .5
        self.subscription_summary: pd.DataFrame | None = None
        self.input_params = InputParamsQuoting(**kwargs)
        self.subscription_manager: None | SubscriptionManager = None
        self.isin_to_ticker = ISIN_TO_TICKER
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        self.activate_driver_price_calculator = kwargs.get("activate_driver_price_calculator", True)
        self.number_of_days = kwargs.get("number_of_days", 5)

        self.theoretical_prices: pd.DataFrame | None = None
        self.book_storage: deque = deque(maxlen=3)
        self.spreads: Optional[pd.Series] = None
        self.strategy_input: pd.DataFrame | pd.Series | None = None
        self.position: Optional[pd.Series] = None
        self.live_weights: Optional[pd.Series] = None
        self.threshold_exceeded_instruments = set()
        self.return_to_publish: list = [1, 2, 3, 4]
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

        self.nav_matrix = pd.DataFrame()
        self.issuer_prices = pd.DataFrame()
        self.weight_nav_matrix = pd.DataFrame()

        # ------------------------------------ STOCK COMPRESSION -------------------------------------------------------
        if self.input_params.isin_nav:
            self.pcf_processor = PCFProcessor(etf_list=self.input_params.isin_nav, date=self.yesterday)
            self.missing_isins = self.pcf_processor.get_missing_pcfs()
            self.classifier = self.pcf_processor.isin_classifier
            self.pcf_controls = PCFControls(self.pcf_processor)
            self.nav_matrix = self.pcf_processor.get_nav_matrix()
            self.weight_nav_matrix = self.pcf_processor.get_nav_matrix(weight=True)
            self.stock_selector = StockSelector(self.weight_nav_matrix)
            stock_to_keep, etf_to_keep = self.stock_selector.truncate_stocks(2000)
            stock_to_keep += list(CURRENCY)
            stock_to_keep = [stock for stock in stock_to_keep if stock not in kwargs.get("stock_to_ignore", [])]
            self.nav_matrix = self.nav_matrix.loc[etf_to_keep, self.nav_matrix.columns.isin(stock_to_keep)]
            self.weight_nav_matrix = self.weight_nav_matrix.loc[etf_to_keep,
            self.weight_nav_matrix.columns.isin(stock_to_keep)]
            self.weight_correction = self.weight_nav_matrix.sum(axis=1)
            self.composition = self.pcf_processor.pcf_composition
            self.composition["PRICE_EUR"] = (self.composition["PRICE_FUND_CCY"] /
                                             self.composition["BSH_ID_ETF"].
                                             map(self.pcf_controls.convert_fund_ccy_to_eur()))
            self.issuer_prices = self.pcf_controls.get_issuer_prices()
        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.instruments_NAV = self.nav_matrix.index.tolist()

        self.instruments_driver = self.input_params.isin_driver
        self.instruments_cluster = self.input_params.isin_cluster
        cluster_price_regressor = list(set(ISINS_ETF_EQUITY).
                                       intersection(self.as_isin(self.input_params.beta_cluster.columns)))

        self.instruments = list(set(self.instruments_NAV + self.instruments_driver
                                    + self.instruments_cluster + cluster_price_regressor))

        self.isins_components = list(set(self.nav_matrix.columns))
        self.securities_list = list(set(self.nav_matrix.columns) | set(self.instruments))

        self.output_theoretical_prices = pd.DataFrame(index=self.instruments_driver)
        self.output_theoretical_prices.index.name = "ETF"

        # self.prices_provider = PricesProviderEquity(etfs=sorted(self.instruments),
        #                                             input_params=self.input_params,
        #                                             subscription_manager=self.subscription_manager,
        #                                             number_of_days=self.number_of_days)
        #
        # self.etf_prices = self.prices_provider.get_hist_prices()
        # self.fx_prices = self.prices_provider.get_hist_fx_prices()
        start, end = today() - self.number_of_days * CustomBDay, today() - CustomBDay

        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\BshDataProvider\config\bshdata_config.yaml")
        self.fx_prices = self.API.market.get_daily_currency(id=[f"EUR{ccy}" for ccy in CURRENCY],
                                                            start=start,
                                                            end=end,
                                                            snapshot_time=time(17),
                                                            fallbacks=[{"source": "bloomberg"}])

        self.etf_prices = self.API.market.get_daily_etf(id=self.instruments,
                                                        start=start,
                                                        end=end,
                                                        snapshot_time=time(17))

        _, fx_full = self.input_params.get_currency_data(self.instruments)
        fx_full = fx_full.reset_index()

        fx_composition = fx_full.pivot(index="index", columns="CURRENCY", values="WEIGHT").fillna(0)
        fx_forward = fx_full.pivot(index="index", columns="CURRENCY", values="WEIGHT_FX_FORWARD").fillna(0)

        fx_forward_needed = fx_forward.columns.tolist()

        fx_forward_prices = self.API.market.get_daily_fx_forward(quoted_currency=fx_forward_needed,
                                                                 start=start,
                                                                 end=end)

        dividends = self.API.info.get_dividends(id=self.instruments)
        ter = self.API.info.get_ter(id=self.instruments)
        ter.update(self.input_params.ter_manual)

        self.adjuster = (
            Adjuster(self.etf_prices)
            .add(TerComponent(ter))
            .add(FxSpotComponent(fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(fx_forward, fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(dividends))
        )

        self.corrected_return: Optional[pd.DataFrame] = pd.DataFrame(index=self.etf_prices.index,
                                                                     columns=self.etf_prices.columns,
                                                                     dtype=float)
        self.bloomberg_subscription_config_path = kwargs.get("bloomberg_subscription_config_path", None)
        self.all_securities = list(set(self.securities_list) | CURRENCY)
        self.all_etf_plus_securities = list(set(self.all_securities + list(ISINS_ETF_EQUITY)))
        self.subscription_manager = SubscriptionManager(self.all_etf_plus_securities,
                                                        self.bloomberg_subscription_config_path)

        self.return_adjustments = self.adjuster.get_adjustments_cumulative()

        # ----------------------------------------- PRICING ------------------------------------------------------------

        # DRIVER
        self.nav = pd.Series(index=self.instruments_NAV)
        self.theoretical_live_driver_price: Optional[pd.Series] = None
        self.theoretical_live_cluster_price: Optional[pd.Series] = None

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

        self.initialize_fx_prices()
        self.initialize_issuer_prices()
        mid = self.market_data.get_mid_eur()
        missing = mid[mid.isna()]
        for instr in missing.index:
            if self.ask_for_stock_drop:
                response = input(f"\nSubscription of {instr}: Want to impute a value? Y/N ").strip().upper() == "Y"
            else:
                response = False
            if response:
                self.impute_user_price(instr)
            else:
                self.nav_matrix.drop(instr, axis=1, inplace=True, errors="ignore")
                self.weight_nav_matrix.drop(instr, axis=1, inplace=True, errors="ignore")

        return True

    def impute_user_price(self, instr):
        while True:
            try:
                price = float(input(f"Enter a price for {instr}: "))
                self.market_data.update(instr, {fld: price for fld in self.market_data.mid_key})
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    def initialize_fx_prices(self):
        """
        Aggiorna i prezzi FX nel market data.
        """
        for ccy, prices in self.fx_prices.items():
            price = prices.dropna()
            if price is not None and not price.empty:
                price = price.iloc[0]
                ccy = str(ccy).replace("EUR", "")
                self.market_data.update(ccy, {fld: price for fld in self.market_data.fields})

    def initialize_issuer_prices(self):
        """
        Aggiorna i prezzi degli strumenti emittenti nel market data.
        """
        mid = self.market_data.get_mid()
        for isin, price in self.issuer_prices.items():
            if not np.isnan(price) and price is not None:
                if isin in self.market_data.securities:
                    ccy = self.market_data.currency_information.get(isin, "EUR")
                    fx = mid.get(ccy, 1)
                    self.market_data.update(str(isin), {fld: price * fx for fld in self.market_data.fields})
                    self.mid_eur[isin] = price

    def update_HF(self):

        self.get_mid()
        self.calculate_cluster_theoretical_price()
        self.calculate_NAV()

        self.gui_redis.export_message("market:nav",
                                      self.round_series_to_tick(self.nav.fillna(0), TICK_SIZE))
        self.gui_redis.export_message("market:nav_bid",
                                      self.round_series_to_tick(self.nav_bid_ask["BID"], TICK_SIZE))
        self.gui_redis.export_message("market:nav_ask",
                                      self.round_series_to_tick(self.nav_bid_ask["ASK"], TICK_SIZE))
        self.gui_redis.export_message("market:theoretical_live_index_cluster_price",
                                      self.round_series_to_tick(self.theoretical_live_index_cluster_price, TICK_SIZE))
        self.gui_redis.export_message("market:theoretical_live_cluster_price",
                                      self.round_series_to_tick(self.theoretical_live_cluster_price.fillna(0),
                                                                TICK_SIZE))
        self.gui_redis.export_message("market:mid",
                                      self.round_series_to_tick(self.mid_eur.fillna(0), TICK_SIZE))

    def update_LF(self):
        # self.check_book_update(1200)
        pass

    def on_book_initialized(self):

        if self.instruments_NAV:
            self.check_live_weights()
            self.check_pcf_controls()
            self.check_inactive_subs()

            self.gui_redis.export_static_data(LIVE_WEIGHTS=self.live_weights,
                                              NAV_WEIGHTS=self.weight_nav_matrix.sum(axis=1),
                                              R2_CLUSTER=self.input_params.r2_cluster)

    def calculate_NAV(self):

        self.nav = (
            (self.nav_matrix @ self.mid_eur[self.nav_matrix.columns]).T.div(
                self.weight_nav_matrix.sum(axis=1)[self.instruments_NAV])
        ).replace({0: np.nan})

        all_book = self.book_eur.loc[self.nav_matrix.columns]

        self.nav_bid_ask = (
            (self.nav_matrix @ all_book).T.div(
                self.weight_nav_matrix.sum(axis=1)[self.instruments_NAV])
        ).replace({0: np.nan}).T

    def get_live_fx_return_correction(self, EUR_CCY: bool = True) -> pd.DataFrame:
        """
        Calculate FX live return correction.
        Returns:
            pd.Series: FX live correction series.
        """
        fx_book: pd.Series = self.market_data.get_mid([ccy for ccy in CURRENCY])  # getting EURCCY
        fx_book.index = fx_book.index.str[-3:]
        fx_live_correction: pd.DataFrame = self.return_adjuster.get_fx_corrections(fx_book,
                                                                                   cumulative=self._cumulative_returns)
        return fx_live_correction

    def get_live_returns(self) -> pd.Series(dtype=float):
        """
        Get live ETF and drivers returns by comparing current prices with historical prices.

        Returns:
            pd.Series: ETF live returns.
        """
        etfs_returns: pd.Series(dtype=float) = self.mid_eur[self.etf_prices.columns] / self.etf_prices - 1
        return etfs_returns.T

    def get_mid(self) -> pd.Series:
        """
        Get the mid-price of book.
        Store corrected returns and a copy of last book

        Returns:
            pd.Series: Series of md-prices for ETFs, Drivers, and FX.
        """

        last_mid = self.market_data.get_mid_eur()
        self.book_eur = self.market_data.get_book_eur()
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

        self.corrected_return = (self.get_live_returns().
                                 add(self.get_live_fx_return_correction().T, fill_value=0).
                                 add(self.return_adjustments.T, fill_value=0))

        # Publish returns
        for i in self.return_to_publish:
            self.gui_redis.export_message(f"market:{i}D_return",
                                          self.corrected_return.iloc[:, -i].astype(float).round(4))

        self.book_storage.append(last_mid)
        return last_mid

    def check_book_update(self, threshold_seconds):

        warning = {isin: (datetime.now() - last_up).seconds
                   for isin, last_up in self.market_data.last_update.items()
                   if (datetime.now() - last_up).seconds > threshold_seconds}
        if warning:
            logging.info(f"no update in the last {threshold_seconds} seconds for {len(warning)} instr:\n" +
                         "\n-".join(f"{isin}: {delay}" for isin, delay in warning.items()))

    def check_live_weights(self):

        weights = self.pcf_processor.get_nav_matrix(weight=True).copy()
        delayed_securities_bool = self.market_data.get_delayed_status()
        self.subscription_summary = pd.DataFrame({isin:
                                                      {"status": status,
                                                       "market": self.subscription_manager.get_ref_market(isin)}
                                                  for isin, status in delayed_securities_bool.items()}).T

        logging.warning("subscription market summary:\n" + self.subscription_summary
                        .groupby("market").
                        agg(count=('status', 'count'),
                            delayed_=('status', 'mean')).
                        to_string())

        live_securities = delayed_securities_bool[~delayed_securities_bool].index.tolist()
        self.live_weights = weights[weights.columns.intersection(live_securities)].sum(axis=1)
        error_live_weights = (self.live_weights[self.live_weights < 0.95]
                              .sort_values(ascending=False)
                              .rename(ISIN_TO_TICKER))
        if len(error_live_weights): logging.warning(f"\nlow live weights for:\n\n" + error_live_weights.to_string())

    def check_pcf_controls(self):
        stock_warnings = self.pcf_controls.check_for_issuers_price_errors()
        if stock_warnings and input("want to drop these stocks?(Y/N):\n\n"
                                    + '\n'.join(stock_warnings)
                                    + "\n\n\n").upper() != "N":
            self.nav_matrix.drop(stock_warnings, axis=1, inplace=True, errors="ignore")
            self.weight_nav_matrix.drop(stock_warnings, axis=1, inplace=True, errors="ignore")
        self.pcf_controls.check_for_my_price_errors(self.market_data.get_mid_eur())
        self.pcf_controls.check_delisting_and_issuer_price(self.market_data.instrument_status)

    def on_stop(self):
        # saving instruments with probably wrong currency

        instr = list(self.threshold_exceeded_instruments)
        (pd.DataFrame({
            'Instrument': instr,
            'ActualCurrency': [self.subscription_manager.get_currency_informations().get(i) for i in instr],
            'ActualMarket': [self.subscription_manager.get_ref_market(i) for i in instr]})
         .to_excel('logging/instruments_threshold_exceeded.xlsx'))

        self.gui_redis.export_static_data(ENGINE_PID=None)

    @property
    def instruments_NAV(self) -> list:
        return self._instruments_NAV

    @instruments_NAV.setter
    def instruments_NAV(self, value: list):
        self._instruments_NAV = [self.as_isin(id) for id in value]

    @instruments_NAV.getter
    def instruments_NAV(self) -> list[str]:
        return self._instruments_NAV

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
            self.theoretical_live_cluster_price = self.cluster_model.get_price_prediction(self.mid_eur,
                                                                                          self.corrected_return.T)
            self.theoretical_live_index_cluster_price = self.index_cluster_model.get_price_prediction(self.mid_eur,
                                                                                                      self.corrected_return.T)
        except Exception as e:
            logging.error(f"Exception occurred while calculating cluster price: {e}")

    def check_inactive_subs(self):

        dump_path = r"C:\AFMachineLearning\Applications\DBAnagrafica\isin_list.txt"
        inactive_isin = [isin for isin, status in self.market_data.instrument_status.items() if status != "ACTV"]
        with open(dump_path, "w") as file:
            for key in inactive_isin:
                file.write(f"{key}\n")

        logging.info(f"Inactive ISIN saved to {dump_path}")

    def check_missing_ccy_info(self, isins):
        dump_path = r"C:\AFMachineLearning\Applications\DBAnagrafica\isin_list_ccy.txt"
        with open(dump_path, "w") as file:
            for key in isins:
                file.write(f"{key}\n")

        logging.info(f"missing ccy ISIN saved to {dump_path}")

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        """ Arrotonda una Series ai tick specificati per ciascun strumento e normalizza i float. """
        ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index]) / 2
        values = series.fillna(0).values.astype(float)
        rounded_values = np.round(np.round(values / ticks) * ticks, 10)
        return pd.Series(rounded_values, index=series.index).fillna(0)
