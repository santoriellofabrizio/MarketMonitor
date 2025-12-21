import logging
from datetime import time
from typing import List
import pandas as pd
from pandas._libs.tslibs.offsets import BDay

from user_strategy.utils import CustomBDay
from user_strategy.utils.Pricing.AggregationFunctions import ForecastAggregator, forecast_aggregation

from user_strategy.utils.InputParamsFI import InputParamsFI

logger = logging.getLogger()


class InputParamsFIQuoting(InputParamsFI):
    """
    This class manages and configures input parameters for Fixed Income ETF analysis.

    Attributes:
        price_snipping_time_string (str | None): A string representing the cutoff time for prices.
        logger (logging.Logger): Logger to track the class execution.
        params (dict): Dictionary containing configuration parameters.
        _TER (pd.DataFrame): DataFrame that holds the Total Expense Ratio (TER) for various ISINs.
        use_cache_ts (bool): Flag indicating whether to use time-series data caching.
        outlier_percentage_NAV (None | float): Threshold percentage for identifying outliers in NAV calculations.
        _YTM_mapping (pd.DataFrame): DataFrame mapping yields to maturity (YTM) to ISINs.
        cluster_anagraphic (pd.DataFrame): Anagraphic data for ISIN clusters.
        _hedge_ratios_cluster (pd.DataFrame): Currency hedging ratios for each ISIN.
        _currency_exposure (pd.DataFrame): Represents the currency exposure for each ISIN.
        isins (List[str]): List of ISINs monitored by the strategy.
        book_storage_size (int | None): Maximum buffer size for historical prices being monitored.
        number_of_days (None | int): Number of days to be used for historical data analysis.
        today (pd.Timestamp): Today's date used as a reference for time-based processing.
        yesterday (datetime.date): Yesterday's date used for daily price change calculations.
        date_from (pd.Timestamp): Start date for historical data analysis.
        price_snipping_time (datetime.time): Cutoff time for acquiring prices.
        min_ctv_to_show_trades (float): Minimum trade value to display trades.
        trade_export_cell (str | None): Reference to the Excel cell for exporting trade data.
        trade_export_sheet (str | None): Name of the Excel sheet for exporting trade data.
        output_trade_columns (List[str] | None): Columns to include in the trade export output_NAV file.
        output_prices_cell (str | None): Excel cell for exporting price data.
        output_prices_sheet (str | None): Excel sheet for exporting price data.
    """

    def __init__(self, params, **kwargs):

        """Initializes the class with the given parameters and sets basic attributes like logger, parameters, and data variables."""

        self._pricing = None
        self.price_snipping_time_string: str | None = "17:00:00"  # Time cutoff for prices in string format
        self.logger = logging.getLogger()  # Logger to track the class activities
        self.params = params  # Configuration parameters passed to the constructor

        self.sql_db_fi_file = None
        self._sql_db_manager = None
        self._oracle_connection = None
        self.Oracle_DB_connection = None
        self._pcf_db_manager = None

        # Attributes for storing data like hedge ratios, currency exposure, etc.
        self.use_cache_ts: bool = True
        self.outlier_percentage_NAV: None | float = None
        self._YTM_mapping: pd.DataFrame = pd.DataFrame()
        self.cluster_anagraphic: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_cluster: pd.DataFrame = pd.DataFrame()
        self._hedge_ratios_drivers: pd.DataFrame = pd.DataFrame()

        self._currency_exposure: pd.DataFrame = pd.DataFrame()
        self._currency_weights: pd.DataFrame = pd.DataFrame()

        self._forecast_aggregator_driver: ForecastAggregator | None = None
        self._forecast_aggregator_cluster: ForecastAggregator | None = None
        self._forecast_aggregator_nav: ForecastAggregator | None = None
        self._forecast_aggregator_brother: ForecastAggregator | None = None
        self.all_instruments: List[str] = []
        self.drivers: pd.DataFrame = pd.DataFrame()
        self.book_storage_size: int | None = None  # Buffer size for storing historical data

        self.number_of_days: None | int = 10  # Number of days for historical data analysis
        self.today: pd.Timestamp = pd.Timestamp.today()  # Today's date
        self.yesterday = (self.today - CustomBDay).date()  # Yesterday's date for daily price calculations
        self._set_config_parameters()  # Call method to set configuration parameters
        self.date_from: pd.Timestamp = self.today - BDay(
            self.number_of_days)  # Calculate start date based on number of days
        h, m, d = map(int, self.price_snipping_time_string.split(":"))  # Parse cutoff time string
        self.price_snipping_time: time = time(hour=h, minute=m, second=d)  # Set cutoff time for acquiring prices
        self.min_ctv_to_show_trades: float = 0  # Minimum trade value for displaying trades
        self.trade_export_cell: str | None = None  # Excel cell for exporting trades
        self.trade_export_sheet: str | None = None  # Excel sheet for exporting trades
        self.output_trade_columns: List[str] | None = None  # Columns to include in trade export
        self.halflife_ewma_cluster: float | None = None  # Placeholder for half-life parameter of EWMA
        self.halflife_ewma_nav: float | None = None
        self.halflife_ewma_driver: float | None = None
        self.output_prices_cell: str | None = None  # Excel cell for exporting prices
        self.output_prices_sheet: str | None = None  # Excel sheet for exporting prices

        self._set_config_parameters()  # Set configuration parameters from provided params
        self._load_inputs()  # Load input data from the Excel file
        self._elaborate_inputs()  # Process loaded inputs
        self.set_forecast_aggregation_func(params["pricing"])

    def _set_config_parameters(self) -> None:
        """
        Sets configuration parameters from keyword arguments.

        """
        for key, value in self.params.items():
            setattr(self, key, value)  # Set attributes dynamically from params

    @property
    def pricing(self):
        return self._pricing

    @pricing.setter
    def pricing(self, kwargs):
        self._pricing = kwargs
        self.set_forecast_aggregation_func(kwargs)

    def set_forecast_aggregation_func(self, kwargs):

        for key in ["cluster", "driver", "nav", "brother"]:
            try:
                params = kwargs[key]
                self.__setattr__(f"_forecast_aggregator_{key}",
                                 forecast_aggregation[params["forecast_aggregation"]](
                                     **params[params["forecast_aggregation"]]))

            except KeyError:
                self.logger.critical(
                    f"forecast aggregator for {key} not implemented. available: {forecast_aggregation}")
                raise KeyboardInterrupt

    @property
    def forecast_aggregator_cluster(self):
        return self._forecast_aggregator_cluster

    @property
    def forecast_aggregator_nav(self):
        return self._forecast_aggregator_nav

    @forecast_aggregator_cluster.setter
    def forecast_aggregator_cluster(self, val):
        self._forecast_aggregator_cluster = val

    @forecast_aggregator_nav.setter
    def forecast_aggregator_nav(self, val):
        self._forecast_aggregator_nav = val

    @property
    def forecast_aggregator_brother(self):
        return self._forecast_aggregator_brother

    @forecast_aggregator_brother.setter
    def forecast_aggregator_brother(self, val):
        self._forecast_aggregator_brother = val

    @property
    def currency_exposure(self) -> pd.DataFrame:
        """Returns the processed currency exposure DataFrame."""
        if (missing_ccy_exposure := self._currency_exposure.index.symmetric_difference(self.etf_isins)).__len__():
            self.logger.critical(f"Missing currency exposure for {', '.join(missing_ccy_exposure)}")
            if input("Do you want to continue? [Y/N] ").lower() != "y": raise KeyError
        return self._currency_exposure

    @property
    def currency_weights(self) -> pd.DataFrame:
        """Returns the processed currency weights DataFrame."""
        return self._currency_weights

    @property
    def forecast_aggregator_driver(self):
        return self._forecast_aggregator_driver

    @forecast_aggregator_driver.setter
    def forecast_aggregator_driver(self, val):
        self._forecast_aggregator_driver = val

    @currency_exposure.setter
    def currency_exposure(self, value: pd.DataFrame) -> None:
        self._currency_exposure = value

    @currency_weights.setter
    def currency_weights(self, value: pd.DataFrame) -> None:
        self._currency_weights = value

    @property
    def YTM_mapping(self):
        return self._YTM_mapping

    @YTM_mapping.setter
    def YTM_mapping(self, value: pd.DataFrame) -> None:
        self._YTM_mapping = value

    @property
    def hedge_ratios_cluster(self):
        return self._hedge_ratios_cluster

    @hedge_ratios_cluster.setter
    def hedge_ratios_cluster(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_cluster = value

    @property
    def hedge_ratios_drivers(self):
        return self._hedge_ratios_drivers

    @hedge_ratios_drivers.setter
    def hedge_ratios_drivers(self, value: pd.DataFrame) -> None:
        value = value.loc[self.etf_isins, self.drivers.index.tolist()]
        self._hedge_ratios_drivers = value

    @property
    def hedge_ratios_brothers(self):
        return self._hedge_ratios_brothers

    @hedge_ratios_brothers.setter
    def hedge_ratios_brothers(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_brothers = value

    @property
    def hedge_ratios_credit_futures_brothers(self):
        return self._hedge_ratios_credit_futures_brothers

    @hedge_ratios_credit_futures_brothers.setter
    def hedge_ratios_credit_futures_brothers(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_credit_futures_brothers = value

    @property
    def hedge_ratios_credit_futures_cluster(self):
        return self._hedge_ratios_credit_futures_cluster

    @hedge_ratios_credit_futures_cluster.setter
    def hedge_ratios_credit_futures_cluster(self, value: pd.DataFrame) -> None:
        self._hedge_ratios_credit_futures_cluster = value
