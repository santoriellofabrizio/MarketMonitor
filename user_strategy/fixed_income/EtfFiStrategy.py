import datetime as dt
from collections import deque
from typing import Optional, Tuple, Union
import pandas as pd
from dateutil.utils import today

from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.utils import CustomBDay
from user_strategy.utils.pricing_models.DataFetching.PricesProviderFI import PricesProviderFI
from user_strategy.utils.pricing_models.NAVBasisCalculator import NAVBasisCalculator
from user_strategy.utils.InputParamsFI import InputParamsFI
from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel, DriverPricingModel
from user_strategy.utils.trade_manager import trade_manager
from user_strategy.utils.bloomberg_subscription_utils.OracleConnection import OracleConnection
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager


class EtfFiStrategy(StrategyUI):
    """
    A class for monitoring fixed income markets, inheriting functionality from MarketMonitorFixedIncomeUI.

    Attributes:
        yesterday_misalignment_cluster (pd.Series): Stores the misalignment of theoretical prices for each ISIN from the
         previous trading day.
        last_export_time (float): Records the last time the trade data was exported,
         measured in seconds since the epoch.
        book_storage (deque): A double-ended queue that holds a fixed number of historical book entries
         (up to `book_storage_size`), used to maintain a short-term record of book prices.
         market data and imputing missing values based on historical prices.
        nav_basis_calculator (NAVBasisCalculator): An instance of the `NAVBasisCalculator` class, used to calculate the
         NAV NAVs.
         using historical prices and adjustments for cluster corrections.
        return_adjustment (float): A cumulative adjustment value based on the results from the data preprocessor.
        market_data: Varies (specific to `EtfFiStrategyInitialized`): Contains market data related to _securities,
         including methods for updating and accessing this data.
        trade_manager: Varies (specific to `EtfFiStrategyInitialized`): Contains trade data, which can be updated with
        new trades and is used for analysis and reporting.
     """

    def __init__(self, *args, **kwargs):
        """
        Initialize the fixed_income class, set up theoretical live prices, and start monitoring.

        Args:
            *args: Additional arguments passed to the superclass.
            **kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.subscription_manager: None | SubscriptionManager = None
        self.corrected_return: pd.DataFrame = pd.DataFrame()
        self.today: dt.date = today().date()
        print(kwargs)
        self.yesterday: dt.date = (today() - CustomBDay).date()
        self.book_mid: pd.DataFrame(dtype=float) | None = None
        self.input_params = InputParamsFI(kwargs)
        self._cumulative_returns: bool = True
        self.bloomberg_subscription_config_path = kwargs.get("bloomberg_subscription_config_path", None)

        # Load the anagraphic data from an Excel file
        self.yesterday_misalignment_cluster: pd.Series = pd.Series(dtype=float)
        self.last_export_time = 0
        self.book_storage: deque = deque(maxlen=self.input_params.book_storage_size)
        self.etf_isins, self.drivers = self.input_params.etf_isins, self.input_params.drivers.index.tolist()
        self.currency_exposure: pd.DataFrame = self.input_params.currency_exposure

        self._all_securities = self.etf_isins + self.currency_exposure.columns.tolist() + self.drivers
        self.subscription_manager = SubscriptionManager(self._all_securities,
                                                        self.bloomberg_subscription_config_path)

        # Initialize the theoretical live price series
        self.theoretical_live_cluster_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                                   name="th live cluster price")
        self.theoretical_live_brother_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                                   name="th live brother price")
        self.theoretical_live_nav_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                               name="th live nav price")
        self.theoretical_live_driver_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                                  name="th live driver price")
        self.theoretical_live_price: pd.DataFrame = pd.DataFrame(dtype=float,
                                                                 index=self.etf_isins,
                                                                 columns=["th live cluster price",
                                                                          "th live nav price",
                                                                          "th live driver price",
                                                                          "th live brother price"])

        self.cluster_correction: pd.Series = self._calculate_cluster_correction(self.input_params.cluster_anagraphic["CLUSTER_ID"])
        self.brothers_correction: pd.Series = self._calculate_cluster_correction(self.input_params.brothers['BROTHER_ID'])

        self.cluster_model = ClusterPricingModel(name="TH PRICING CLUSTER",
                                                 beta=self.input_params.hedge_ratios_cluster,
                                                 returns=self.corrected_return,
                                                 forecast_aggregator=self.input_params.forecast_aggregator_cluster,
                                                 cluster_correction=self.cluster_correction)

        self.driver_model = DriverPricingModel(beta=self.input_params.hedge_ratios_drivers,
                                               returns=self.corrected_return,
                                               forecast_aggregator=self.input_params.forecast_aggregator_driver)

        self.brother_model = ClusterPricingModel(name = "TH PRICING BROTHER",
                                                 beta=self.input_params.hedge_ratios_brothers,
                                                 returns=self.corrected_return,
                                                 forecast_aggregator=self.input_params.forecast_aggregator_brother,
                                                 cluster_correction=self.brothers_correction)

        self.trade_manager = TradeManager(book_storage=self.book_storage, input_params=self.input_params)

        self.on_start_strategy()

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        """

        self.market_data.securities = self._all_securities
        self.market_data.subscription_dict = self.subscription_manager.get_subscription_dict()
        yesterday_price = pd.concat([self.historical_prices.loc[self.yesterday],
                                     self.historical_fx.loc[self.yesterday]])
        for isin, price in yesterday_price.iteritems():
            self.market_data.update(isin, {field: price for field in self.market_data.mid_key}, perform_check=False)

    def on_start_strategy(self) -> None:
        """
        Start monitoring by fetching historical prices, impute missing values, and set up data preprocessing.
        """
        self.price_provider = PricesProviderFI(self.etf_isins, self.input_params, self.subscription_manager)
        self.currency_exposure: pd.DataFrame = self.input_params.currency_exposure
        self.historical_prices: pd.DataFrame = self.price_provider.get_hist_prices()
        self.historical_fx: pd.DataFrame = self.price_provider.get_hist_fx_prices()

        # Set up the NAV NAVs calculator
        self.nav_basis_calculator: NAVBasisCalculator = NAVBasisCalculator(
            OracleConnection(),
            self.historical_prices,
            self.historical_fx,
            self.input_params
        )

        # Calculate theoretical relative return NAV
        self.theoretical_misalignment_basis: pd.Series(dtype=float) = self.nav_basis_calculator.get_basis_misalignment()
        self.NAVs: pd.DataFrame = self.nav_basis_calculator.get_NAVs()

        self.return_adjustments = self.price_provider.get_adjustments(cumulative=self._cumulative_returns)

    def on_trade(self, new_trades: pd.DataFrame) -> None:

        """
        Update the trades DataFrame with new trades data.

        Args:
            new_trades (pd.DataFrame): DataFrame containing new trade data.
        """
        self.trade_manager.on_trade(new_trades)
        self.export_data(
            {"data": self.trade_manager.get_gui_output(),
             "cell": self.input_params.trade_export_cell,
             "sheet": self.input_params.trade_export_sheet,
             "force": False})

    def update_HF(self, *args, **kwargs) -> Union[dict, Tuple]:
        """
        Update prices over time. Time interval is set from config. Whatever is returned is displayed in the gui.

        Returns:
            Optional[tuple]: A tuple containing the theoretical live price, output_NAV cell, and sheet names.
        """
        self.get_mid()
        self.calculate_theoretical_prices()
        self.export_data({"data": dt.datetime.now(),
                          "cell": "A6",
                          "sheet": "PriceContainer",
                          "force": False})

        return {"data": (pd.concat([self.theoretical_live_cluster_price,
                                    self.theoretical_live_nav_price,
                                    self.theoretical_live_driver_price,
                                    self.theoretical_live_brother_price], axis=1)),
                "cell": self.input_params.output_prices_cell,
                "sheet": self.input_params.output_prices_sheet,
                "force": False}

    def calculate_theoretical_prices(self):
        self.theoretical_live_driver_price =\
        self.driver_model.get_price_prediction(self.book_mid,
                                                self.corrected_return.T,
                                                self.input_params.forecast_aggregator_driver)
        self.theoretical_live_cluster_price =\
            self.cluster_model.get_price_prediction(self.book_mid,
                                                    self.corrected_return.T,
                                                    self.input_params.forecast_aggregator_cluster,
                                                    self.cluster_correction)

        self.theoretical_live_brother_price =\
            self.brother_model.get_price_prediction(self.book_mid,
                                                    self.corrected_return.T,
                                                    self.input_params.forecast_aggregator_brother,
                                                    self.brothers_correction)

        all_predictions_NAV = (self.book_mid[self.etf_isins]
                               * (1 + self.theoretical_misalignment_basis.mul(self.cluster_correction))
                               * (1 + self.cluster_model.last_misalignment_cluster))
        theoretical_nav_prices = self.input_params.forecast_aggregator_nav(all_predictions_NAV)

        self.theoretical_live_nav_price.update(theoretical_nav_prices)

    def get_live_fx_return_correction(self) -> pd.Series:
        """
        Calculate FX live return correction.
        Returns:
            pd.Series: FX live correction series.
        """
        fx_book: pd.Series = self.book_mid[self.input_params.currencies_EUR_ccy]
        fx_live_correction: pd.Series = self.price_provider.get_fx_correction(fx_book, cumulative=self._cumulative_returns)
        return fx_live_correction.T

    def get_live_returns(self) -> pd.Series(dtype=float):
        """
        Get live ETF and drivers returns by comparing current prices with historical prices.

        Returns:
            pd.Series: ETF live returns.
        """
        etfs_returns: pd.Series(dtype=float) = self.book_mid / self.historical_prices - 1
        return etfs_returns.T

    @staticmethod
    def _calculate_cluster_correction(cluster_anagraphic: pd.Series) -> pd.Series:
        """
        Calculate the cluster correction factor for each subcluster.

        Returns:
            pd.Series: Series with correction factors for each ISIN.
        """
        cluster_sizes: pd.Series = cluster_anagraphic.value_counts()

        # Compute weight: (n-1)/n if n > 1, else 1
        correction = cluster_anagraphic.map(lambda x: (cluster_sizes[x] - 1) / cluster_sizes[x] if cluster_sizes[x] > 1 else 1)
        return correction

    def get_mid(self) -> pd.Series:
        """
        Get the mid-price of book.
        Store corrected returns and a copy of last book

        Returns:
            pd.Series: Series of mid-prices for ETFs, Drivers, and FX.
        """

        self.book_mid = (last_book := self.market_data.get_mid())
        # self.corrected_return = (self.get_live_returns().
        #                          add(self.get_live_fx_return_correction(), fill_value=0).
        #                          add(self.return_adjustments, fill_value=0))
        # self.corrected_return = (self.get_live_returns().
        #                          add(self.return_adjustments, fill_value=0))
        self.corrected_return = self.get_live_returns()

        self.book_storage.append(last_book)
        return last_book

    def wait_for_book_initialization(self):
        self.logger.info("Checking all subscription started")
        return True
