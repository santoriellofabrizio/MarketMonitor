import logging
import warnings
from datetime import date
from typing import Tuple

import pandas as pd
from dateutil.utils import today

from pandas.errors import PerformanceWarning
from xbbg.blp import bdh, bdp

from user_strategy.FixedIncomeETF import memoryFixedIncome
from user_strategy.utils import CustomBDay
from user_strategy.utils.InputParamsFI import InputParamsFI
from user_strategy.utils.bloomberg_subscription_utils.OracleConnection import OracleConnection
from user_strategy.utils.Pricing.ExcelStoringDecorator import save_to_excel


class NAVBasisCalculator:

    def __init__(self, oracle_connection: OracleConnection,
                 etf_prices: pd.DataFrame, currency_prices: pd.DataFrame,
                 input_params_FI: InputParamsFI):

        self.oracle_connection: OracleConnection = oracle_connection
        self.outlier_perc = input_params_FI.outlier_percentage_NAV
        self.isins = input_params_FI.etf_isins
        self.hedge_ratios = input_params_FI.hedge_ratios_cluster.loc[self.isins, self.isins]
        self.etf_prices = etf_prices.copy()
        self.currency_prices = currency_prices.copy()
        self.start_date: date = etf_prices.index.min()
        self.end_date: date = etf_prices.index.max()
        self.logger = logging.getLogger()
        self.misalignment: pd.Series = pd.Series(index=self.isins, name="theoretical_basis_misalignment")
        self.basis: pd.DataFrame = pd.DataFrame(index=self.isins)
        self.NAVs: pd.DataFrame = pd.DataFrame(index=self.isins)
        self.start_calculations()

    def _fetch_nav_oracle(self, oracle_connection) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Fetch NAV data from Oracle and identify missing ISINs.

        :param oracle_connection: OracleConnection instance
        :return: Tuple of NAV dataframe and list of missing ISINs
        """
        NAV_info_Oracle = oracle_connection.get_nav_daily(self.isins, self.start_date)
        NAV_Oracle = NAV_info_Oracle.pivot_table(index="REF_DATE", columns="BSH_ID", values="NAV")
        NAV_ccy_oracle = NAV_info_Oracle.groupby("BSH_ID").first()["NAV_CCY"]

        missing_isins = [isin for isin in self.isins if isin not in NAV_Oracle.columns]
        if missing_isins:
            self.logger.info(f"WARNING: {','.join(missing_isins)} are missing from Oracle data.")

        return NAV_Oracle, NAV_ccy_oracle[NAV_Oracle.columns], missing_isins

    def _fetch_nav_bloomberg(self, missing_isins: tuple) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fetch NAV and NAV currency data from Bloomberg for missing ISINs.

        :param missing_isins: List of missing ISINs
        :return: Tuple of NAV and NAV currency dataframes
        """
        navs, ccy_info = self._fetch_cached_nav_bloomberg(missing_isins, self.start_date, self.end_date)
        if (missing_info_bbg := [isin for isin in missing_isins if isin not in navs.columns]).__len__():
            self.logger.info(f"WARNING: {','.join(missing_info_bbg)} NAVs are not available in Bloomberg or Oracle data.")
            navs[missing_info_bbg] = None

        if (missing_info_bbg_ccy := [isin for isin in missing_isins if isin not in ccy_info.index]).__len__():
            self.logger.info(f"WARNING: {','.join(missing_info_bbg_ccy)} NAVs ccy are not available in Bloomberg or Oracle data.")
            ccy_info[missing_info_bbg_ccy] = None
        return navs, ccy_info

    @save_to_excel("adjusted_prices_for_fx")
    def _adjust_prices_for_currency(self, prices: pd.DataFrame, NAV_ccy: pd.Series, currencies: pd.DataFrame):
        """
        Adjust prices for currency conversion based on NAV currency.

        :param prices: Price dataframe
        :param NAV_ccy: NAV-Currency of NAV dataframe
        :param currencies: Currency conversion dataframe
        """
        for isin, ccy in NAV_ccy.items():
            if ccy == "EUR": continue
            try:
                prices[isin] *= currencies[f"EUR{ccy}"]
            except KeyError:
                print(f"Warning: Missing currency data for {isin}, skipping conversion.")

        return prices

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches NAV and prices from Oracle and Bloomberg sources, processing any missing ISINs.

        :return: Tuple of prices and NAV dataframes
        """
        # Initialize Oracle and Bloomberg connections

        # Fetch NAV data from Oracle
        NAV_Oracle, NAV_ccy_oracle, missing_isins = self._fetch_nav_oracle(self.oracle_connection)

        # Fetch NAV data from Bloomberg for missing ISINs
        NAV_BBG, NAV_ccy_BBG = self._fetch_nav_bloomberg(tuple(missing_isins))

        # Combine NAV data from Oracle and Bloomberg
        NAV = self._combine_nav_data(NAV_Oracle, NAV_BBG)
        NAV_ccy_info = self._combine_NAV_ccy_info(NAV_ccy_oracle, NAV_ccy_BBG)

        # Download currency and price data
        currencies, prices = self.currency_prices, self.etf_prices

        # Adjust prices for currency exchange rates
        prices = self._adjust_prices_for_currency(prices, NAV_ccy_info, currencies)

        # prices, NAV = self._deal_with_missing_values(prices, NAV)

        return prices, NAV

    @save_to_excel("NAV_values")
    def _combine_nav_data(self, NAV_Oracle: pd.DataFrame, NAV_BBG: pd.DataFrame) -> pd.DataFrame:
        """
        Combine NAV data from Oracle and Bloomberg, with forward/backward filling for missing data.

        :param NAV_Oracle: NAV data from Oracle
        :param NAV_BBG: NAV data from Bloomberg
        :return: Combined NAV dataframe
        """
        NAV_BBG.index = pd.to_datetime(NAV_BBG.index)
        NAV_Oracle.index = pd.to_datetime(NAV_Oracle.index)

        # Combine NAV data from Bloomberg and Oracle
        NAV = pd.concat([NAV_BBG, NAV_Oracle], axis=1)
        NAV.index = [i.date() for i in NAV.index]
        for ticker, percent in NAV_Oracle.isna().mean().items():
            if percent > 0: self.logger.info(f"{ticker} has {percent*100:.0f}% of N/A (NAV)")
        return NAV

    @save_to_excel("NAV_ccy info")
    def _combine_NAV_ccy_info(self,  NAV_ccy_oracle: pd.Series, NAV_ccy_BBG: pd.Series) -> pd.Series:
        NAV_ccy = pd.concat([NAV_ccy_oracle, NAV_ccy_BBG])
        return NAV_ccy

    def start_calculations(self):
        prices, self.NAVs = self.get_data()
        self.basis = self.calculate_basis(prices, self.NAVs)
        available_isin = [isin for isin in self.basis.columns if not self.basis[isin].isna().any()]
        betas = self.hedge_ratios.loc[available_isin, available_isin].T
        yesterday = (today() - CustomBDay).date()
        real_basis_return = self.basis.loc[yesterday, available_isin] - self.basis.loc[:yesterday, available_isin]
        theoretical_basis_returns = real_basis_return @ betas
        self.misalignment = real_basis_return - theoretical_basis_returns
        missing_isin = [isin for isin in self.isins if isin not in available_isin]
        if missing_isin.__len__():
            self.logger.info(f"Theoretical return for {', '.join(missing_isin)}) is assumed to be 0")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PerformanceWarning)
                self.misalignment[missing_isin] = 0

    def get_basis_misalignment(self):
        return self.misalignment

    def get_NAVs(self):
        return self.NAVs

    @staticmethod
    def _deal_with_missing_values(prices: pd.DataFrame, NAV: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for name, df in (("NAV", prices), ("price", NAV)):
            missing = df.index[df.isna().any(axis=1)].tolist()
            if len(missing):
                print(f"Discarding {missing} from NAV analysis since {name} are missing")
                prices.drop(missing, inplace=True, errors="ignore")
                NAV.drop(missing, inplace=True, errors="ignore")
        return prices, NAV

    @staticmethod
    @memoryFixedIncome.cache
    def _fetch_cached_nav_bloomberg(missing_isins: tuple, start_date: date, end_date: date)  -> Tuple[pd.DataFrame, pd.Series]:
            """
            Fetch NAV and NAV currency data from Bloomberg for missing ISINs.

            :param missing_isins: List of missing ISINs
            :return: Tuple of NAV and NAV currency dataframes
            """
            missing_isins_bbg_subs = [f"{isin} IM EQUITY" for isin in missing_isins]
            NAV_BBG = bdh(missing_isins_bbg_subs,
                          flds="FUND_NET_ASSET_VAL",
                          start_date=start_date,
                          end_date=end_date)
            NAV_BBG.columns = [col.split(" ")[0] for col in NAV_BBG.columns.droplevel(1)]
            NAV_ccy_BBG = bdp(tickers=missing_isins_bbg_subs, flds="nav_crncy")["nav_crncy"]
            NAV_ccy_BBG.index = [idx.split(" ")[0] for idx in NAV_ccy_BBG.index]

            return NAV_BBG, NAV_ccy_BBG

    @save_to_excel("Basis")
    def calculate_basis(self, prices, NAV):
        return (prices/NAV - 1).ffill().bfill()



