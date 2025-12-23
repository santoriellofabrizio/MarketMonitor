import logging
from datetime import time
from typing import Optional, List

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from user_strategy.utils import CustomBDay, memoryPriceProvider
from user_strategy.utils.Pricing.DataFetching.download_functions import download_daily_prices_fx, \
    process_downloaded_prices, download_daily_prices, get_price_for_day_time
from user_strategy.utils.Pricing.ExcelStoringDecorator import save_to_excel
from user_strategy.utils.InputParams import InputParams
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager

logger = logging.getLogger()


class PricesProvider:

    def __init__(self, etfs, input_params: InputParams = None, subscription_manager: SubscriptionManager = None,
                 instruments_to_download_eod: List[str] = None, additional_contracts: pd.DataFrame = None,
                 trading_currency: pd.DataFrame = None, **kwargs):
        """
        Initialize the PricesProvider with ETF data and input parameters.

        Args:
            etfs: List of ETFs to process.
            input_params (InputParams, optional):
             An object containing configuration parameters. such that:
            - currencies_EUR_ccy: list of currencies to download like: EURUSD, EURAUD ecc...
            - price_snipping_time: (time) time of price snipping
            - use_cache_ts: boolean indicating whether to use existing cached data or not
            - number of days: int indicating the number of business days of data
            **kwargs: Additional keyword arguments to override or provide parameters not specified in InputParams.
        """
        # -------------------------- INPUT PARAMS ------------------------------------------------------
        self.etfs = etfs
        self._subscription_manager = subscription_manager
        self._input_params: InputParams = input_params
        if self._input_params is None: self._input_params = {}
        self.additional_contracts = additional_contracts if additional_contracts is not None else pd.DataFrame()
        self.trading_currency = trading_currency if trading_currency is not None else pd.DataFrame()

        # Extract parameters from `self._input_params` if provided, else use defaults
        if isinstance(self._input_params, InputParams):
            self.currencies_EUR_ccy = getattr(self._input_params, 'currencies_EUR_ccy', kwargs.get('currencies_EUR_ccy', []))
            self.price_snipping_time: time = getattr(self._input_params, 'price_snipping_time',
                                                     kwargs.get('price_snipping_time', time(17)))
            self.drivers_anagraphic = getattr(self._input_params, 'drivers', kwargs.get('drivers', pd.DataFrame()))
            self.index_anagraphic = getattr(self._input_params, 'index_data', kwargs.get('index_data', pd.DataFrame()))
            self.use_cache_ts: bool = getattr(self._input_params, 'use_cache_ts', kwargs.get('use_cache_ts', True))
            self.number_of_days: int = getattr(self._input_params, 'number_of_days', kwargs.get('number_of_days', 22))
            self.ytm_mapping: pd.DataFrame = getattr(self._input_params, 'YTM_mapping', kwargs.get('YTM_mapping', pd.DataFrame()))
            self.currency_weights: pd.DataFrame = getattr(self._input_params, 'currency_weights', kwargs.get('currency_weights', pd.DataFrame()))
            self.ter_manual = getattr(self._input_params, 'ter_manual', kwargs.get('ter_manual', pd.DataFrame()))
        elif isinstance(self._input_params, dict):
            self.currencies_EUR_ccy = self._input_params.get('currencies_EUR_ccy', kwargs.get('currencies_EUR_ccy', []))
            self.price_snipping_time: time = self._input_params.get('price_snipping_time',
                                                     kwargs.get('price_snipping_time', time(17)))
            self.drivers_anagraphic = self._input_params.get('drivers', kwargs.get('drivers', pd.DataFrame()))
            self.index_anagraphic = self._input_params.get('index_data', kwargs.get('index_data', pd.DataFrame()))
            self.use_cache_ts: bool = self._input_params.get('use_cache_ts', kwargs.get('use_cache_ts', True))
            self.number_of_days: int = self._input_params.get('number_of_days', kwargs.get('number_of_days', 22))
            self.ytm_mapping: pd.DataFrame = self._input_params.get('YTM_mapping', kwargs.get('YTM_mapping', pd.DataFrame()))
            self.currency_weights: pd.DataFrame = self._input_params.get('currency_weights', kwargs.get('currency_weights', pd.DataFrame()))
            self.ter_manual: pd.DataFrame = self._input_params.get('ter_manual', kwargs.get('ter_manual', pd.DataFrame()))

        if instruments_to_download_eod is None:
            self.instruments_to_download_eod = []
        else:
            self.instruments_to_download_eod = instruments_to_download_eod

        # -------------------------- INITIALIZATION ------------------------------------------------------
        self.logger = logging.getLogger()
        self.historical_fx: pd.DataFrame = pd.DataFrame()
        self.historical_prices: pd.DataFrame = pd.DataFrame()

        # Clear cache if needed
        if not self.use_cache_ts:
            memoryPriceProvider.clear()

        # Date calculations
        self.today: pd.Timestamp = pd.Timestamp.today()
        self.yesterday = (self.today - CustomBDay).date()
        self.date_from: pd.Timestamp = self.today - CustomBDay * self.number_of_days
        self.date_range = [d.date() for d in pd.bdate_range(start=self.date_from,
                                                            end=self.today - CustomBDay,
                                                            freq=CustomBDay)]

        # Load initial data
        self.load_data()
        self._return_adjuster = self._instantiate_return_adjuster()
        self._return_adjuster.download_data()

    def _instantiate_return_adjuster(self):
        return None

    def load_data(self):
        self.historical_fx = self.get_hist_fx_prices_from_oracle()
        self.historical_prices = pd.concat([self.get_hist_etf_prices_from_oracle(),
                                            self.get_hist_generic_instr_prices_from_oracle(
                                                drivers_anagraphic=pd.concat([self.drivers_anagraphic, self.index_anagraphic, self.additional_contracts]))],
                                           axis=1)
        self._impute_missing_values_from_bbg()


    def get_adjustments(self, cumulative: bool = False) -> pd.DataFrame:
        return self._return_adjuster.get_adjustments(cumulative)

    def get_fx_correction(self, fx_update: Optional[pd.Series] = None, cumulative: bool = False):
        fx_update.index = fx_update.index.str[-3:]
        return self._return_adjuster.get_fx_corrections(fx_update, cumulative)

    def get_hist_prices(self) -> pd.DataFrame:
        return self.historical_prices

    def get_hist_fx_prices(self) -> pd.DataFrame:
        return self.historical_fx

    @save_to_excel("prices ETF")
    def get_hist_etf_prices_from_oracle(self) -> DataFrame:
        """
        Fetch historical ETF prices from Oracle.
        """
        if not self.etfs: return pd.DataFrame()
        prices = self._download_prices(self.etfs, "ETFs from Oracle")
        return process_downloaded_prices(prices, self.date_range, col_name="isin")

    @save_to_excel("prices drivers")
    def get_hist_generic_instr_prices_from_oracle(
            self,
            market_isins_dict: dict | None = None,
            drivers_anagraphic: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical instrument prices from Oracle.

        Args:
            market_isins_dict (dict, optional): A dictionary mapping markets to lists of instrument ISINs.
            drivers_anagraphic (pd.DataFrame, optional): A dataframe with instrument details, including:
                - index: Instrument ID.
                - "BLOOMBERG_CODE": Subscription of bloomberg. Optional, will be used in case of N/A
                - "PRICE_SOURCE_MARKET": Associated market.
                - "MARKET_CODE": code in associated market

        Returns:
            pd.DataFrame: Historical prices for instruments.
        """
        if not market_isins_dict:
            if drivers_anagraphic is None or drivers_anagraphic.empty:
                return pd.DataFrame()
            market_isins_dict = {
                market: group["MARKET_CODE"].dropna().tolist()
                for market, group in drivers_anagraphic.groupby("PRICE_SOURCE_MARKET") if market
            }

        prices = [
            download_daily_prices(self.date_range, isins, self.price_snipping_time, market=market,
                                  desc=f"Downloading driver prices from Oracle {market}")
            for market, isins in market_isins_dict.items()
        ]
        prices = process_downloaded_prices(pd.concat(prices, ignore_index=True), self.date_range, col_name="isin")

        if drivers_anagraphic is not None:
            prices.rename(
                columns={val: key for key, val in drivers_anagraphic["MARKET_CODE"].items() if
                         val in prices.columns},
                inplace=True
            )
            for instr in drivers_anagraphic.index:
                if instr not in prices.columns:
                    prices[instr] = None

        return prices

    @save_to_excel("prices FX")
    def get_hist_fx_prices_from_oracle(self) -> pd.DataFrame:
        """
        Fetch historical FX prices from Oracle.
        """
        if not self.currencies_EUR_ccy.__len__(): return pd.DataFrame()
        closing_price_currencies = download_daily_prices_fx(self.date_range, self.currencies_EUR_ccy,
                                                            self.price_snipping_time)

        return process_downloaded_prices(closing_price_currencies,
                                         array_date=self.date_range,
                                         instruments=self.currencies_EUR_ccy,
                                         col_name="currency_pair")

    def _download_prices(self, tickers: list, description: str) -> DataFrame:
        """
        Download prices based on the provided tickers and date range.

        Args:
            tickers (list): List of ticker symbols.
            description (str): Description for progress tracking.

        Returns:
            DataFrame: The downloaded price data.
        """
        return download_daily_prices(self.date_range, tickers, self.price_snipping_time,
                                     desc=f"Downloading {description}")

    def _impute_missing_values_from_bbg(self) -> None:
        """
        Fill NA values in the DataFrame with prices fetched from Bloomberg.
        """
        # Count missing prices in historical data
        for date in self.date_range:
            if date not in self.historical_prices.index.values:
                if not self.historical_prices.empty: self.historical_prices.loc[date] = None
            if date not in self.historical_fx.index.values:
                if not self.historical_fx.empty: self.historical_fx.loc[date] = None

        if (num_of_missing := self._count_missing_prices()) == 0: return

        if input(f"{num_of_missing:.0f} prices are missing. Download from BBG? Y/N: ").strip().upper() != "Y":
            return

        # Gather missing entries for both historical prices and FX
        missing_entries = self._gather_missing_entries()

        # Create a progress bar for imputation
        for df, _date, ticker in tqdm(missing_entries, desc="Downloading missing prices from BBG", unit="entry"):
            self.logger.info(f"Imputing price for {ticker}-{_date} from Bloomberg")
            subscription = self.get_bbg_subscription(ticker)
            download_eod = False
            if ticker in self.instruments_to_download_eod:
                download_eod = True
            price = get_price_for_day_time(subscription, _date, self.price_snipping_time, download_eod)
            df.loc[_date, ticker] = price  # Update the DataFrame with the fetched price
        for df in [self.historical_prices, self.historical_fx]:
            df.sort_index(ascending=False, inplace=True, axis=0)
            if missing := df.loc[:, df.isna().any(axis=0)].columns.tolist():
                if input(f"Still missing:\n" + "\n".join(missing) + "\nWant to bfill? (Y/N)").strip().upper() == "Y":
                    df.bfill(inplace=True)
                    df.ffill(inplace=True)
            assert df.isna().sum().sum() == 0

    def _count_missing_prices(self) -> int:
        """
        Count the number of missing prices in historical data.
        """
        return (self.historical_prices.isna().sum().sum() +
                self.historical_fx.isna().sum().sum())

    def _gather_missing_entries(self) -> list:
        """
        Gather all missing entries in historical data.
        """
        return [(df, date_index, ticker)
                for df in [self.historical_prices, self.historical_fx]
                for date_index in df.index.values.tolist()
                for ticker in df.columns
                if pd.isna(df.loc[date_index, ticker])]

    def get_bbg_subscription(self, ticker: str) -> str:
        """
        Get Bloomberg subscription string for the given ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            str: Subscription string for the ticker.
        """
        return self._subscription_manager.get_subscription_string(ticker)


if __name__ == "__main__":
    price_provider = PricesProvider(["IE00B02KXL92"])
    a = price_provider.get_hist_prices()
    b   = 0