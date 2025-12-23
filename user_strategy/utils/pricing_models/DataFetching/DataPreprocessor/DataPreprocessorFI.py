from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_strategy.utils.pricing_models.DataFetching.DataPreprocessor.DataPreprocessor import fx_forward_mapping, \
    DataPreprocessor
from user_strategy.utils.pricing_models.DataFetching.download_functions import download_yas, download_repo, \
    download_fx_forward_composition
from user_strategy.utils.InputParamsFI import InputParamsFI
from user_strategy.utils.pricing_models.ExcelStoringDecorator import save_to_excel


class DataPreprocessorFI(DataPreprocessor):

    def __init__(self,
                 prices: pd.DataFrame,
                 fx_prices: pd.DataFrame,
                 inputs: InputParamsFI | None = None,
                 **kwargs):
        super().__init__(prices, fx_prices, inputs, **kwargs)
        self.ytm_inputs = getattr(inputs, 'YTM_mapping', kwargs.get('ytm_inputs'))

    @save_to_excel("carry")
    def _adjust_carry(self) -> DataFrame:
        """
        Adjust prices for carry based on yield-to-maturity (YTM) data.
        :return: DataFrame containing the YTM carry adjustment.
        """
        # Map ISINs to yield-to-maturity data
        yas_mapping: pd.Series = self.ytm_inputs.get("mapping_subscription", pd.Series()).dropna()
        hard_coded_isin: pd.Series = self.ytm_inputs["hard_coding"].dropna()
        repo_mapping: pd.Series = self.ytm_inputs["repo_rate"].dropna()

        # Download YTM data from Bloomberg (or other sources)
        YTM: pd.Series = download_yas([isin for isin in sorted(self.isins) if (isin in yas_mapping and
                                                                               isin not in hard_coded_isin.index)],
                                      yas_mapping)

        self._adjust_carry_future(YTM, repo_mapping, hard_coded_isin.index.tolist())

        # Apply hardcoded YTM values if available
        for isin, yas in hard_coded_isin.items():
            YTM[isin] = float(yas) / 100

        # Calculate year fractions
        year_fractions = self._get_year_fractions()
        # Calculate YTM adjustment (YTM * year fraction)
        YTM_adjustment = pd.DataFrame(np.outer(YTM.values, year_fractions.values), columns=year_fractions.index,
                                      index=YTM.index)
        return - YTM_adjustment.fillna(0)

    def _adjust_carry_future(self, YTM: pd.Series, repo_mapping: pd.Series, hard_coded_isin: list) -> pd.Series:
        REPO: pd.Series = download_repo([isin for isin in sorted(self.isins) if (isin in repo_mapping and
                                                                                 isin not in hard_coded_isin)],
                                        repo_mapping)
        for isin, repo in REPO.items():
            YTM[isin] -= repo
        return YTM

    @save_to_excel("fx_frwd")
    def _adjust_fx_forward_carry(self) -> DataFrame:
        """
        Adjust prices for FX forward carry based on forward contract data.
        :return: DataFrame containing the FX forward carry adjustment.
        """
        currencies = self.currency_exposure.columns  # List of currency pairs
        dates = self.prices.index.tolist()  # Dates in the prices DataFrame

        # Download FX forward prices
        fx_forward_list = [fx_forward_mapping[fx] for fx in currencies]  # Map currency pairs to forward tickers
        print(fx_forward_list)
        prices_fx_frwd = download_fx_forward_composition(fx_forward=fx_forward_list, date_range=dates)
        prices_fx_frwd["EURJPY1M Curncy"] *= 100  # Scale for JPY contracts
        prices_fx_frwd *= (12 / 10000)  # Adjust for 1M contract, convert from NAVs points

        # Rename columns with original currency pair codes
        prices_fx_frwd.columns = prices_fx_frwd.columns.map({val: key for key, val in fx_forward_mapping.items()})

        # Calculate FX forward carry correction
        year_fractions = self._get_year_fractions()
        fx_carry_correction = (self.currency_exposure @ (prices_fx_frwd / self.fx_prices[currencies]).T
                               * year_fractions).mul(self.currency_hedged, axis=0)

        return fx_carry_correction

    def calculate_adjustments(self) -> Dict[str, DataFrame]:
        """
        Calculate all required adjustments: dividends, TER, YTM, and FX forward carry.
        :return: Dictionary containing all adjustments.
        """
        return {
            "dividend_adjustment": self._adjust_dividends(all_or_last="all"),
            "ter_adjustments": self._adjust_ter(),
            "ytm_adjustments": self._adjust_carry(),
            "fx_frwd_adjustments": self._adjust_fx_forward_carry()
        }
