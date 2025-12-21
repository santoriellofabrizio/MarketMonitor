import logging
from collections import deque
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.utils import today
import datetime as dt

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from UserStrategy.utils import CustomBDay
from UserStrategy.utils.Pricing.DataFetching.PricesProviderFI import PricesProviderFI
from UserStrategy.utils.InputParamsFIQuoting import InputParamsFIQuoting
from UserStrategy.utils.Pricing.NAVBasisCalculator import NAVBasisCalculator
from UserStrategy.utils.Pricing.PricingModel import ClusterPricingModel, DriverPricingModel, \
    CreditFuturesCalendarSpreadPricingModel, CreditFuturesInterestRatePricingModel
from UserStrategy.utils.Pricing.TheoreticalPriceManager import TheoreticalPriceManager
from UserStrategy.utils.Pricing.IRPManager import IRPManager
from market_monitor.utils.SubscriptionManager.SubscriptionManager import SubscriptionManager
from market_monitor.utils.enums import TICK_SIZE


class EtfFiPriceEngine(StrategyUI):
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

        self.subscription_manager: None | SubscriptionManager = None
        self.corrected_returns: pd.DataFrame = pd.DataFrame()
        self.today: dt.date = today().date()
        self.yesterday: dt.date = (today() - CustomBDay).date()
        self.book_mid: pd.DataFrame(dtype=float) | None = None
        self.input_params = InputParamsFIQuoting(kwargs)
        self._cumulative_returns: bool = True
        self.bloomberg_subscription_config_path = kwargs.get("bloomberg_subscription_config_path", None)

        # Load the anagraphic data from an Excel file
        self.yesterday_misalignment_cluster: pd.Series = pd.Series(dtype=float)
        self.last_export_time = 0
        self.book_storage: deque = deque(maxlen=self.input_params.book_storage_size)
        self.etf_isins = self.input_params.etf_isins
        self.drivers_data = self.input_params.drivers
        self.drivers_list = self.drivers_data.index.to_list()

        self.credit_futures_contracts_data = self.input_params.credit_futures_data
        self.credit_futures = self.credit_futures_contracts_data['INSTRUMENT'].unique().tolist()
        self.credit_futures_contracts = self.credit_futures_contracts_data.index.tolist()
        self.index_drivers = self.input_params.index_data.index.to_list()
        cutoff_date = max(self.credit_futures_contracts_data['EXPIRY_DATE']) + dt.timedelta(days=97)    # 3 months and 1 week

        self.irs_data = self.input_params.irs_data
        self.irs_contracts_list = self.irs_data.index.to_list()
        self.irp_data = self.input_params.irp_data
        self.irp_manager = IRPManager(cutoff_date, self.irp_data, self.irs_data.loc[~self.irs_data.index.isin(["ESTR3M", "SOFR3M"])])
        self.irp_contracts_data = self.irp_manager.get_contracts_list_data()
        self.irp_contracts_list = self.irp_contracts_data.index.to_list()

        self.currency_exposure: pd.DataFrame = self.input_params.currency_exposure
        self.trading_currency: pd.DataFrame = self.input_params.trading_currency
        self.fx_list = self.currency_exposure.columns.tolist()
        self._all_securities = (self.etf_isins + self.fx_list + self.drivers_list + self.credit_futures_contracts +
                                self.irp_contracts_list)
        self.subscription_manager = SubscriptionManager(self._all_securities,
                                                        self.bloomberg_subscription_config_path)

        self.cluster_correction: pd.Series = self._calculate_cluster_correction_2(self.input_params.hedge_ratios_cluster)
        self.brothers_correction: pd.Series = self._calculate_cluster_correction_2(self.input_params.hedge_ratios_brothers)

        # Initialize the theoretical live price object
        self.theoretical_price_manager = TheoreticalPriceManager()
        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live cluster price",
                                                   instruments=self.etf_isins,
                                                   model=ClusterPricingModel(name="th live cluster price",
                                                                             beta=self.input_params.hedge_ratios_cluster,
                                                                             returns=self.corrected_returns,
                                                                             forecast_aggregator=self.input_params.forecast_aggregator_cluster,
                                                                             cluster_correction=self.cluster_correction
                                                                             )
                                                   )
        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live brother price",
                                                   instruments=self.etf_isins,
                                                   model=ClusterPricingModel(name="th live brother price",
                                                                             beta=self.input_params.hedge_ratios_brothers,
                                                                             returns=self.corrected_returns,
                                                                             forecast_aggregator=self.input_params.forecast_aggregator_brother,
                                                                             cluster_correction=self.brothers_correction
                                                                             ),
                                                   )
        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live driver price",
                                                   instruments=self.etf_isins,
                                                   model=DriverPricingModel(name="th live driver price",
                                                                            beta=self.input_params.hedge_ratios_drivers,
                                                                            returns=self.corrected_returns,
                                                                            forecast_aggregator=self.input_params.forecast_aggregator_driver
                                                                            ),
                                                   )
        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live cluster credit futures price",
                                                   instruments=self.credit_futures_contracts + self.index_drivers,
                                                   model=ClusterPricingModel(name="th live cluster credit futures price",
                                                                             beta=self.input_params.hedge_ratios_credit_futures_cluster.loc[self.credit_futures_contracts + self.index_drivers],
                                                                             returns=self.corrected_returns,
                                                                             forecast_aggregator=self.input_params.forecast_aggregator_cluster,
                                                                             disable_warning=True),
                                                   )
        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live brother credit futures price",
                                                   instruments=self.credit_futures_contracts,
                                                   model=ClusterPricingModel(name="th live brother credit futures price",
                                                                             beta=self.input_params.hedge_ratios_credit_futures_brothers.loc[self.credit_futures_contracts],
                                                                             returns=self.corrected_returns,
                                                                             forecast_aggregator=self.input_params.forecast_aggregator_brother,
                                                                             disable_warning=True),
                                                   )
        self.credit_futures_proxy = pd.DataFrame(index=self.credit_futures_contracts, columns=["Future Proxy", "IRP"])
        self.credit_futures_proxy["Future Proxy"] = self.credit_futures_contracts_data['PREVIOUS_CONTRACT']
        self.credit_futures_contracts_data['REGION'] = self.credit_futures_contracts_data['REGION'].replace("", "US")
        self.credit_futures_proxy["IRP"] = self.credit_futures_contracts_data.merge(self.irp_data.reset_index(), left_on="REGION", right_on="REGION")['INSTRUMENT_ID'].to_list()
        self.credit_futures_proxy['Expiry'] = self.credit_futures_contracts_data['EXPIRY_DATE']
        self.credit_futures_proxy['Proxy Expiry'] = self.credit_futures_contracts_data['PREVIOUS_CONTRACT_EXPIRY_DATE']

        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live spread credit futures price",
                                                   instruments=self.credit_futures_contracts,
                                                   model=CreditFuturesCalendarSpreadPricingModel(name="th live spread credit futures price",
                                                                                         target_variables=self.credit_futures_contracts,
                                                                                         variables_proxy=self.credit_futures_proxy,
                                                                                         irp_manager=self.irp_manager
                                                                                        )
                                                   )
        self.theoretical_price_manager.add_pricing(dtype=float,
                                                   name="th live ir credit futures price",
                                                   instruments=self.credit_futures_contracts,
                                                   model=CreditFuturesInterestRatePricingModel(name="th live ir credit futures price",
                                                                                               target_variables=self.credit_futures_contracts,
                                                                                               variables_proxy=self.credit_futures_proxy[['IRP', 'Expiry']],
                                                                                               irp_manager=self.irp_manager
                                                                                               )
                                                   )
        self.gui_redis = RedisMessaging()
        self.on_start_strategy()

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        """
        self.market_data.set_securities(self._all_securities, "market")
        self.book_mid = pd.Series(index=self._all_securities, dtype=float)
        self.market_data.currency_information = self.subscription_manager.get_currency_informations()
        for id, subscription_string in self.subscription_manager.get_subscription_dict().items():
            if id in self.irp_contracts_list + self.irs_contracts_list:
                fields = ["LAST_PRICE"]
            else:
                fields = ["BID", "ASK"]

            self.market_data.subscribe_bloomberg(
                id=id,
                subscription_string=subscription_string,
                fields=fields,
                params={"interval": 1}
            )
        for currency in self.fx_list:
            self.market_data.subscribe_bloomberg(
                id=id,
                subscription_string=f"{currency} Curncy",
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

        yesterday_price = pd.concat([self.historical_prices.loc[self.yesterday],
                                     self.historical_fx.loc[self.yesterday]])
        for isin, price in yesterday_price.items():
            if isin in self._all_securities:
                self.market_data.update(isin, {field: price for field in self.market_data.mid_key}, perform_check=False)

    def on_start_strategy(self) -> None:
        """
        Start monitoring by fetching historical prices, impute missing values, and set up data preprocessing.
        """
        relevant_columns = ['BLOOMBERG_CODE', 'PRICE_SOURCE_MARKET', 'MARKET_CODE']
        additional_contracts = pd.concat([self.credit_futures_contracts_data[relevant_columns], self.irp_contracts_data[relevant_columns]])
        self.prices_provider = PricesProviderFI(etfs=self.etf_isins,
                                                input_params=self.input_params,
                                                subscription_manager=self.subscription_manager,
                                                instruments_to_download_eod=self.index_drivers + self.irs_contracts_list + self.irp_contracts_list,
                                                additional_contracts=additional_contracts,
                                                trading_currency=self.trading_currency)
        self.historical_prices: pd.DataFrame = self.prices_provider.get_hist_prices()
        self.historical_fx: pd.DataFrame = self.prices_provider.get_hist_fx_prices()

        self.irp_manager.save_historical_prices(self.historical_prices)


        # Set up the NAV NAVs calculator
        # self.nav_basis_calculator: NAVBasisCalculator = NAVBasisCalculator(
        #     OracleConnection(),
        #     self.historical_prices,
        #     self.historical_fx,
        #     self.input_params
        # )

        # Calculate theoretical relative return NAV
        # self.theoretical_misalignment_basis: pd.Series(dtype=float) = self.nav_basis_calculator.get_basis_misalignment()
        # self.NAVs: pd.DataFrame = self.nav_basis_calculator.get_NAVs()

        self.return_adjustments = self.prices_provider.get_adjustments(cumulative=self._cumulative_returns)

    def update_HF(self, *args, **kwargs) -> Union[dict, Tuple]:
        """
        Update prices over time. Time interval is set from config. Whatever is returned is displayed in the gui.

        Returns:
            Optional[tuple]: A tuple containing the theoretical live price, output_NAV cell, and sheet names.
        """
        self.get_mid()
        self.calculate_theoretical_prices()

        pm = self.theoretical_price_manager
        rtt = self.round_series_to_tick
        now = dt.datetime.now().isoformat()

        self.gui_redis.export_message(
            "th_live_cluster_price",
            rtt(pm.get_theoretical_prices("th live cluster price"), TICK_SIZE),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message(
            "th_live_driver_price",
            rtt(pm.get_theoretical_prices("th live driver price"), TICK_SIZE),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message(
            "th_live_brother_price",
            rtt(pm.get_theoretical_prices("th live brother price"), TICK_SIZE),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message(
            "th_live_credit_futures_cluster_price",
            pm.get_theoretical_prices("th live cluster credit futures price"),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message(
            "th_live_credit_futures_brother_price",
            pm.get_theoretical_prices("th live brother credit futures price"),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message(
            "th_live_credit_futures_spread_price",
            pm.get_theoretical_prices("th live spread credit futures price").dropna(),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message(
            "th_live_credit_futures_ir_price",
            pm.get_theoretical_prices("th live ir credit futures price").dropna(),
            skip_if_unchanged=True
        )

        self.gui_redis.export_message("mid", self.book_mid, skip_if_unchanged=True)
        self.gui_redis.export_message("time_now", now)

    def calculate_theoretical_prices(self):
        self.theoretical_price_manager.calculate_theorical_prices(self.book_mid, self.corrected_returns)

    def get_live_fx_return_correction(self) -> pd.DataFrame:
        """
        Calculate FX live return correction.
        Returns:
            pd.Series: FX live correction series.
        """
        fx_book: pd.Series = self.book_mid[self.input_params.currencies_EUR_ccy]
        fx_live_correction: pd.DataFrame = self.prices_provider.get_fx_correction(fx_book, cumulative=self._cumulative_returns)
        return fx_live_correction

    def get_live_returns(self) -> pd.Series(dtype=float):
        """
        Get live ETF and drivers returns by comparing current prices with historical prices.

        Returns:
            pd.Series: ETF live returns.
        """
        all_returns: pd.Series(dtype=float) = self.book_mid / self.historical_prices - 1
        return all_returns.T

        self.market_data.get_delayed_status()

    def stop(self):
        pass

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

    @staticmethod
    def _calculate_cluster_correction_2(cluster_betas: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
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
        cluster_threshold: pd.Series = threshold/(cluster_betas!=0).sum(axis=1)
        # here we count only the beta which are above the threshold
        cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1)+1
        # the correction is than calculated as the number of elements which truly influence our calculations
        correction = cluster_sizes.where(cluster_sizes == 1, (cluster_sizes-1) / cluster_sizes)
        return correction

    def get_mid(self) -> pd.Series:
        """
        Get the mid-price of book.
        Store corrected returns and a copy of last book

        Returns:
            pd.Series: Series of mid-prices for ETFs, Drivers, and FX.
        """

        if self.book_mid is not None:
            last_book = self.market_data.get_data_field(field = ["BID", "ASK"], securities=[sec for sec in self._all_securities if sec not in self.irp_contracts_list + self.irs_contracts_list])
            last_price_ir = self.market_data.get_data_field(field = ["LAST_PRICE"], securities=self.irp_contracts_list + self.irs_contracts_list)
            last_bid = last_book["BID"].replace({0: np.nan})
            last_ask = last_book["ASK"].replace({0: np.nan})
            spread = last_ask / last_bid - 1
            if len(missing_book := spread[spread.isna()].index):
                logging.warning(f"bid is zero for {', '.join(missing_book)}")

            is_outlier = (last_bid.isna() | last_ask.isna() | (spread > 0.015)) & (last_book.index.isin(self.etf_isins))
            last_valid_book = last_book[~is_outlier]
            self.book_mid.update(last_valid_book.mean(axis=1))
            self.book_mid.update(last_price_ir)
        else:
            self.book_mid = self.market_data.get_mid()
        self.corrected_returns = (self.get_live_returns().
                                  add(self.get_live_fx_return_correction().T, fill_value=0).
                                  add(self.return_adjustments.T, fill_value=0))

        self.book_storage.append(self.book_mid)
        return self.book_mid

    def wait_for_book_initialization(self):
        logging.info("Checking all subscription started")
        return True

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        """ Arrotonda una Series ai tick specificati per ciascun strumento e normalizza i float. """
        ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index])
        values = series.fillna(0).values.astype(float)
        rounded_values = np.round(np.round(values / ticks) * ticks, 10)
        return pd.Series(rounded_values, index=series.index).fillna(0)