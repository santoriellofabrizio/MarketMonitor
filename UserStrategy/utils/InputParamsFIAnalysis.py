import logging
from typing import List
import pandas as pd
from UserStrategy.utils import CustomBDay
from UserStrategy.utils.InputParamsFI import InputParamsFI

logger = logging.getLogger()


class InputParamsFIAnalysis(InputParamsFI):
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

        """Initializes the class with the given parameters and sets basic
         attributes like logger, parameters, and data variables."""

        super().__init__(params, **kwargs)
        self.params = params  # Configuration parameters passed to the constructor

        self.book_storage_size: int | None = None  # Buffer size for storing historical data

        self.today: pd.Timestamp = pd.Timestamp.today()  # Today's date
        self.yesterday = (self.today - CustomBDay).date()  # Yesterday's date for daily price calculations
        self._set_config_parameters()  # Call method to set configuration parameters
        self.min_ctv_to_show_trades: float = 0  # Minimum trade value for displaying trades
        self.trade_export_cell: str | None = None  # Excel cell for exporting trades
        self.trade_export_sheet: str | None = None  # Excel sheet for exporting trades
        self.output_trade_columns: List[str] | None = None  # Columns to include in trade export
        self.output_prices_cell: str | None = None  # Excel cell for exporting prices
        self.output_prices_sheet: str | None = None  # Excel sheet for exporting prices

        self._set_config_parameters()  # Set configuration parameters from provided params
        self._load_inputs()  # Load input data from the Excel file
        self._elaborate_inputs()  # Process loaded inputs

    def _set_config_parameters(self) -> None:
        """
        Sets configuration parameters from keyword arguments.

        """
        for key, value in self.params.items():
            setattr(self, key, value)  # Set attributes dynamically from params

    def _elaborate_inputs(self) -> None:
        """Elaborates the inputs by processing the currency exposure and hedge information."""
        self.all_instruments = self.etf_isins + self.drivers.index.tolist()

