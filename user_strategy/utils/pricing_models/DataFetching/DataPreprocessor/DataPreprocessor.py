import logging
from typing import  Dict, Optional

import numpy as np
import pandas as pd
from dateutil.utils import today
from pandas import DataFrame, Series

from user_strategy.utils import CustomBDay
from user_strategy.utils.pricing_models.ExcelStoringDecorator import save_to_excel
from user_strategy.utils.InputParams import InputParams


# Configure pandas to avoid warnings about downcasting
pd.set_option('future.no_silent_downcasting', True)

# FX forward ticker mappings (currency pairs to Bloomberg tickers)
fx_forward_mapping: Dict[str, str] = {
    "EURUSD": "EUR1M BGN Curncy",
    "EURGBP": "EURGBP1M Curncy",
    "EURCNH": "EURCNH1M Curncy",
    "EURJPY": "EURJPY1M Curncy",
    "EURAUD": "EURAUD1M Curncy",
    "EURCAD": "EURCAD1M Curncy",
    "EURCHF": "EURCHF1M Curncy",
    "EURINR": "EURINR1M Curncy",
    "EURIDR": "EURIDR1M Curncy",
    "EURRON": "EURRON1M Curncy",
    "EURMXN": "EURMXN1M Curncy",
    "EURZAR": "EURZAR1M Curncy",
    "EURKRW": "EURKRW1M Curncy",
    "EURCOP": "EURCOP1M Curncy",
    "EURCZK": "EURCZK1M Curncy",
    "EURMYR": "EURMYR1M Curncy",
    "EURPLN": "EURPLN1M Curncy",
    "EURTHB": "EURTHB1M Curncy",
    "EURBRL": "EURBRL1M Curncy",
}

logger = logging.getLogger()


class DataPreprocessor:
    """
    Class responsible for preprocessing price data, handling adjustments
    like dividends, TER, YTM, and FX forward carry for ETFs.
    """

    def __init__(self,
                 prices: pd.DataFrame,
                 fx_prices: pd.DataFrame,
                 inputs: InputParams | None = None,
                 **kwargs):
        """
        Initialize the data processor with prices, FX rates, and optional parameters.

        :param prices: DataFrame containing the prices of instruments (columns) for each datetime (rows).
        :param fx_prices: DataFrame containing historical FX rates.
        :param inputs: Object containing external inputs like YTM_mapping, TER, currency exposure, etc.
        :param kwargs: Additional keyword arguments for optional parameters.
        """
        if not isinstance(prices, pd.DataFrame):
            raise TypeError('Please use a DataFrame for prices.')

        # Ensure the index is datetime formatted
        try:
            prices.index = pd.to_datetime(prices.index).date
        except AttributeError as e:
            logging.error("Error: Please use a Datetime index for the prices DataFrame")
            raise TypeError("Invalid index format") from e

        # Initialize mandatory parameters
        self.isins = prices.columns
        self.prices = prices
        self.fx_prices = fx_prices
        self.number_of_days = (prices.index.max() - prices.index.min()).days if len(prices.index) > 1 else 0

        self.ter_inputs = getattr(inputs, 'TER', kwargs.get('ter_inputs'))

        self.currency_exposure = getattr(inputs, 'currency_exposure', kwargs.get('currency_exposure'))
        self.currency_hedged = getattr(inputs, 'currency_hedged', kwargs.get('currency_hedged'))
        self.snipping_time_fx_frwd = getattr(inputs, 'price_snipping_time', kwargs.get('snipping_time_fx_frwd', None))

    def calculate_adjustments(self, **kwargs) -> Dict[str, DataFrame]:
        """
        Calculate all required adjustments: dividends, TER, YTM, and FX forward carry.
        :return: Dictionary containing all adjustments.
        """
        return {
            "dividend_adjustment": self._adjust_dividends(**kwargs),
            "ter_adjustments": self._adjust_ter()
        }

    @save_to_excel("dividends")
    def _adjust_dividends(self, market_of_instruments: Optional[Dict[str, str]] = None, all_or_last='last') -> DataFrame:
        """
        Adjust prices for dividends by downloading dividend data and converting them based on FX rates.
        :param market_of_instruments: Optional dictionary mapping ISINs to specific markets.
        :return: DataFrame containing the dividend adjustment.
        """
        if market_of_instruments is None: market_of_instruments = {}

        dividends = download_dividends(sorted(self.isins), n_days=self.number_of_days,
                                       market_of_instruments=market_of_instruments,
                                       all_or_last='last')

        try:
            self.dividends_ccy = download_dividends_currency(self.isins.unique().tolist())

        except Exception as e:
            logger.error(f"Error while downloading dividends currency: {e}")


        # Filter dividends by the ex-dividend date (must be within the price data range)
        dividends = dividends.loc[
            (dividends["ex_date"] > self.prices.index.min()) & (dividends["ex_date"] <= today().date())]

        # Initialize an empty DataFrame for dividend adjustments
        self.dividend_adjustment = pd.DataFrame(0.00, columns=self.prices.index, index=self.prices.columns)

        # Calculate the dividend adjustments for each instrument
        for isin, row in dividends.iterrows():
            dividend_currency = self.dividends_ccy[isin]  # Get the currency of the dividend
            ex_date = row["ex_date"]  # Ex-dividend date
            # FX rate conversion (if currency is not EUR)
            if dividend_currency == "GBp":
                fx = self.fx_prices.loc[(ex_date - CustomBDay).date(), f"EURGBP"]/100
            else:
                fx = self.fx_prices.loc[(ex_date - CustomBDay).date(), f"EUR{dividend_currency}"] if dividend_currency != "EUR" else 1
            isin = isin.split(" ")[0]  # Strip any extraneous data from ISIN
            dividend = row["dividend_amount"] / fx  # Convert dividend amount to EUR
            price_etf = self.prices.loc[(ex_date - CustomBDay).date(), isin]  # Previous day's price for return calculation
            dividend_return_adjustment = dividend / price_etf  # Calculate the dividend return adjustment

            # Set the adjustment in the DataFrame
            self.dividend_adjustment.at[isin, (ex_date - CustomBDay).date()] = dividend_return_adjustment

        self.dividend_adjustment.fillna(0, inplace=True)
        return self.dividend_adjustment.cumsum(axis=1)

    @save_to_excel("ter")
    def _adjust_ter(self) -> DataFrame:
        """
        Adjust prices for Total Expense Ratio (TER) of instruments.
        :return: DataFrame containing the TER adjustment.
        """
        # Calculate the fraction of the year for each date
        ter_inputs = self.ter_inputs if isinstance(self.ter_inputs, pd.DataFrame) else  self.ter_inputs or []
        year_fractions = self._get_year_fractions()

        # Download TER data for instruments that don't have hardcoded values
        TER = download_ter([isin for isin in self.isins if isin not in ter_inputs])/100

        # Overwrite TER values with hardcoded ones (if provided)
        if "HARD CODING" in ter_inputs:
            if isinstance(ter_inputs, pd.DataFrame):
                TER.update(pd.Series(ter_inputs["HARD CODING"].dropna()), )
            elif isinstance(ter_inputs, dict):
                TER.update(ter_inputs, )

        # Calculate TER adjustment (TER * year fraction)
        TER_adjustment = pd.DataFrame(np.outer(TER.values, year_fractions.values), columns=year_fractions.index,
                                      index=TER.index)
        return TER_adjustment

    def _get_year_fractions(self) -> Series:
        """
        Calculate the year fractions for each date based on business days.
        :return: Series of year fractions for each date in the price index.
        """
        date_deltas = (
                self.prices.index.to_series().diff().fillna(pd.Timedelta(days=0)) / pd.Timedelta(days=365)).cumsum()
        return -date_deltas.fillna(0)
