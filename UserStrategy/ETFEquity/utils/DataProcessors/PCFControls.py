import logging
from datetime import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd

from UserStrategy.ETFEquity.utils.DataProcessors.PCFProcessor import PCFProcessor
from UserStrategy.utils import CustomBDay
from market_monitor.utils.enums import ISIN_TO_TICKER, ALL_ETF_ISINS

logger = logging.getLogger()


class PCFControls:

    def __init__(self, pcf_processor):
        self.max_allowed_threshold_my_prices = 0.1
        self.max_allowed_return_of_components = 0.1
        self.pcf_processor = pcf_processor

        self.securities = self.pcf_processor.get_securities()
        self.etf = [isin for isin in self.securities if isin in ALL_ETF_ISINS]
        self.components = [isin for isin in self.securities if isin not in ALL_ETF_ISINS]

        self.issuer_prices_data = pcf_processor.get_issuer_prices()
        self.add_price_in_eur()

    def check_old_ref_date(self) -> Optional[list[str]]:
        yesterday = datetime.today() - CustomBDay
        etf_to_ref_date = self.pcf_processor.get_etf_to_ref_date()
        old_pcfs = etf_to_ref_date[etf_to_ref_date < yesterday.normalize()]
        if not old_pcfs.empty:
            if input(f"some PCF are not updated\n: {old_pcfs.to_string()} \n want to drop them? (Y/N").upper() == "Y":
                return old_pcfs.index.tolist()
            else:
                return None
    def check_for_issuers_price_errors(self) -> list:
        prices_difference = (self.issuer_prices_data
                             .groupby("BSH_ID_COMP")["PRICE_EUR"]
                             .apply(lambda x: np.abs(x.dropna().max() / x.dropna().min()) - 1)).dropna()

        prices_warnings = prices_difference[prices_difference > self.max_allowed_return_of_components]

        if len(prices_warnings):

            price_warning_comp = ((self.issuer_prices_data.loc[self.issuer_prices_data["BSH_ID_COMP"].isin(prices_warnings.index)]
                                   .pivot_table(columns="BSH_ID_COMP",
                                                index="BSH_ID_ETF",
                                                values="PRICE_EUR"))).rename(ISIN_TO_TICKER)
            logger.warning(f"Return of these components exceeded {self.max_allowed_return_of_components*100:.2f}%\n"+
                           price_warning_comp.to_string() + "\n")

            return prices_warnings.index.tolist()

    def check_delisting_and_issuer_price(self, instrument_status):

        self.instruments_status = instrument_status
        for instr, status in self.instruments_status.items():
            if status not in ["ACTV","BAD_SEC"] :
                if instr in self.pcf_processor.get_issuer_prices().index:
                    if isinstance(self.pcf_processor.issuer_prices[instr], (pd.DataFrame, pd.Series)):
                        price = self.pcf_processor.issuer_prices[instr].mean()
                    else:
                        price = self.pcf_processor.issuer_prices[instr]
                    logger.warning(f"Instrument {instr} seems UNACTIVE,"
                                        f" but issuer price is {price:.3f}")
    def check_for_my_price_errors(self, my_prices: Union[pd.DataFrame, pd.Series])->pd.DataFrame:

        issuer_prices = self.issuer_prices_data.groupby("BSH_ID_COMP")["PRICE_EUR"].mean()
        valid_issuer_prices = issuer_prices[(issuer_prices > 0) & (~issuer_prices.isna())]
        check_errors = np.abs(my_prices / valid_issuer_prices - 1) > self.max_allowed_threshold_my_prices
        if (n := check_errors.sum()) > 0:
            logger.warning(f"\nPrice Difference issuer-book exceeded threshold for {n} instruments:")

            errors = check_errors.index[check_errors].unique()
            weights = self.pcf_processor.get_nav_matrix(weight=True)
            prices = pd.DataFrame({
                "my_prices": my_prices[errors],
                "issuer_prices_data": issuer_prices[errors],
                "return": (my_prices[errors] / issuer_prices[errors] - 1) * 100,
                "max component isin": weights[errors].idxmax(),
                "max component ticker": weights[errors].idxmax().map(ISIN_TO_TICKER),
                "max component weight(%)": weights[errors].max() * 100,
            })
            prices["error_effect on price (BP)"] = prices["max component weight(%)"] * prices["return"]
            prices.sort_values(by="error_effect on price (BP)", key=lambda x: x.abs(), inplace=True)
            logger.warning("\n" + prices.to_string() + "\n")
            return prices

    def perform_checks(self, my_prices: Union[pd.DataFrame, pd.Series] = None) -> None:
        stock_warnings = self.check_for_issuers_price_errors()
        prices = self.check_for_my_price_errors(my_prices)

    def convert_fund_ccy_to_eur(self):

        self.nav_currency_info = self.pcf_processor.get_etf_nav_ccy(self.etf)
        assert self.nav_currency_info.groupby("ISIN").agg({"NAV_CCY": "nunique"}).max().max() == 1
        fund_ccy = self.nav_currency_info.set_index("ISIN")["NAV_CCY"]
        fund_ccy = fund_ccy.loc[~fund_ccy.index.duplicated(keep='first')]
        currencies_to_download = fund_ccy.drop("EUR", errors="ignore").unique()
        from xbbg.blp import bdp
        currencies_to_download = currencies_to_download[currencies_to_download != "EUR"]
        if len(currencies_to_download) > 0:
            yesterday_closing_prices = bdp([f"EUR{ccy} curncy"
                                            for ccy in currencies_to_download[currencies_to_download != "EUR"]],
                                           ["px_last_eod"])["px_last_eod"]
            yesterday_closing_prices.index = [idx[3:6] for idx in yesterday_closing_prices.index]
        else:
            yesterday_closing_prices = pd.Series()
        yesterday_closing_prices["EUR"] = 1
        fx_correction = fund_ccy.map(yesterday_closing_prices)
        return fx_correction.loc[[etf for etf in self.etf if etf in fx_correction]]

    def add_price_in_eur(self):
        fx_correction = self.convert_fund_ccy_to_eur()
        self.issuer_prices_data.loc[:, "PRICE_EUR"] = (self.issuer_prices_data["PRICE_FUND_CCY"] /
                                                self.issuer_prices_data["BSH_ID_ETF"].map(fx_correction))

    def get_issuer_prices(self) -> pd.Series:
        return self.issuer_prices_data.set_index("BSH_ID_COMP")["PRICE_EUR"].dropna()

if __name__ == "__main__":

    pcf_prc = PCFProcessor(etf_list=["LU0322253906"])
    pcf_cntrls = PCFControls(pcf_prc)
    pcf_cntrls.perform_checks()
    a = 0
