import datetime as dt
from collections import deque
from typing import Optional, Tuple, Union
import pandas as pd
from dateutil.utils import today

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.utils import CustomBDay
from user_strategy.utils.InputParamsFIAnalysis import InputParamsFIAnalysis
from user_strategy.utils.TradeManager.trade_manager import TradeManager


class FIAnalysis(StrategyUI):
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
        Initialize the FixedIncomeETF class, set up theoretical live prices, and start monitoring.

        Args:
            *args: Additional arguments passed to the superclass.
            **kwargs: Additional keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.today: dt.date = today().date()
        self.yesterday: dt.date = (today() - CustomBDay).date()
        self.book_mid: pd.DataFrame(dtype=float) | pd.Series(dtype=float) = pd.Series(dtype=float)
        self.input_params = InputParamsFIAnalysis(kwargs)
        self.book_storage: deque = deque(maxlen=3)

        # Load the anagraphic data from an Excel file
        self.yesterday_misalignment_cluster: pd.Series = pd.Series(dtype=float)
        self.last_export_time = 0
        self.etf_isins, self.drivers = self.input_params.etf_isins, self.input_params.drivers.index.tolist()
        self.credit_futures_contracts_data = self.input_params.credit_futures_data
        self.credit_futures_contracts = self.credit_futures_contracts_data.index.tolist()
        self.credit_futures_tickers = {
            c: f"{c[:4]}_{dt.datetime.strptime(c[4:], '%Y%m').strftime('%b').upper()}{c[6:8]}" for c in
            self.credit_futures_contracts}
        self.credit_futures_tickers_reversed = {v: k for k, v in self.credit_futures_tickers.items()}

        self.price_multiplier = self.input_params.price_multiplier

        # Initialize the theoretical live price series
        self.theoretical_live_cluster_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                                   name="th live cluster price")
        self.theoretical_live_nav_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                               name="th live nav price")
        self.theoretical_live_driver_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                                  name="th live driver price")
        self.theoretical_live_brother_price: pd.Series = pd.Series(dtype=float, index=self.etf_isins,
                                                                   name="th live brother price")
        self.theoretical_live_credit_futures_cluster_price: pd.Series = pd.Series(dtype=float,
                                                                                  index=self.credit_futures_contracts,
                                                                                  name="th live cluster credit futures price")
        self.theoretical_live_credit_futures_brother_price: pd.Series = pd.Series(dtype=float,
                                                                                  index=self.credit_futures_contracts,
                                                                                  name="th live brother credit futures price")

        self.trade_manager = TradeManager(self.book_storage,
                                          self.theoretical_live_brother_price,
                                          **kwargs["trade_manager"])

        self.redis_publisher = RedisMessaging()

        self.on_start_strategy()

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        """

        all_securities = self.etf_isins + self.drivers + self.credit_futures_contracts
        self.market_data.set_securities(all_securities)
        subscription_manager = self.market_data.get_subscription_manager()
        for channel in ["th_live_nav_price", "th_live_driver_price", "th_live_cluster_price", "th_live_brother_price",
                        "th_live_credit_futures_cluster_price", "th_live_credit_futures_brother_price", "mid"]:
            subscription_manager.subscribe_redis(channel=channel, store="market")

    def on_trade(self, new_trades: pd.DataFrame) -> None:
        """
        Update the trades DataFrame with new trades data.

        Args:
            new_trades (pd.DataFrame): DataFrame containing new trade data.
        """
        new_trades['isin'] = pd.Series(new_trades.index.map(self.credit_futures_tickers_reversed),
                                       index=new_trades.index).fillna(new_trades['isin'])
        new_trades['price_multiplier'] = [
            self.price_multiplier.loc[isin, 'CONTRACT_SIZE'] if isin in self.price_multiplier.index else 1
            for isin in new_trades['isin']
        ]

        self.trade_manager.on_trade(new_trades)

        quoting_trades = self.trade_manager.get_trades_from_isin(self.etf_isins, 200)
        quoting_trades_cf = self.trade_manager.get_trades_from_isin(self.credit_futures_contracts, 200)

        all_trades = self.trade_manager.get_filtered_trades(n=20)

        self.redis_publisher.export_message(channel="trades", value=all_trades)


    def update_HF(self, *args, **kwargs) -> Union[dict, Tuple]:
        """
        Update prices over time. Time interval is set from config. Whatever is returned is displayed in the gui.

        Returns:
            Optional[tuple]: A tuple containing the theoretical live price, output_NAV cell, and sheet names.
        """
        self.get_live_data()

    def get_live_data(self):
        """
        Get the mid-price of book.
        Store corrected returns and a copy of last book

        Returns:
            pd.Series: Series of mid-prices for ETFs, Drivers, and FX.
        """

        (self.theoretical_live_nav_price
        .update(self.market_data.get_data_field(
            "theoretical_live_nav_price")))
        (self.theoretical_live_driver_price
        .update(self.market_data.get_data_field(
            "theoretical_live_driver_price")))
        (self.theoretical_live_cluster_price
        .update(
            self.market_data.get_data_field("theoretical_live_cluster_price")))
        (self.theoretical_live_brother_price
        .update(self.market_data.get_data_field(
            "theoretical_live_brother_price")))
        (self.book_mid
        .update(
            last_book := self.market_data.get_data_field("mid")))

        self.book_storage.append((dt.datetime.now(), pd.Series(last_book).copy()))

    def wait_for_book_initialization(self):
        return True
