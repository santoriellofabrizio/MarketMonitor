import logging
from typing import Optional

import pandas as pd
import datetime as dt

from pandas._libs.tslibs.offsets import BDay


from user_strategy.equity.utils.DataProcessors.Classifier import Classifier
from user_strategy.equity.utils.SQLUtils.PCFDBManager import PCFDBManager
from user_strategy.utils.bloomberg_subscription_utils.OracleConnection import memoryPCF
from user_strategy.utils.enums import Instrument, ISINS_ETF_EQUITY, ISIN_TO_TICKER


class PCFProcessor:
    """
    Processor for Portfolio Composition File (PCF) data, primarily for ETFs.
    Handles data retrieval, transformation, and validation from an Oracle database.
    """

    def __init__(self, etf_list, logger: logging.Logger = None, date: Optional[dt.date] = None, **kwargs):
        """
        Initializes the PCFProcessor.

        """
        self.nav_matrix = pd.DataFrame()
        self.cash_components_matrix = None
        self.etf_list = etf_list
        self._date = date
        self.kwargs = kwargs
        self.logger = logger or logging.getLogger()
        self.pcf_manager = PCFDBManager(cache_bool=True)
        self.today = dt.datetime.now().date()
        self.isin_classifier = Classifier()
        if self.etf_list is not None and len(self.etf_list) > 0:
            self.start_processor()

    def start_processor(self):
        """
        Performs the initial setup and data retrieval for the processor.
        """
        self.logger.info("....Starting retrieving pcf composition..")
        self.retrieve_pcf_composition()
        self.logger.info(".... retrievied pcf composition!")
        self.logger.info("....Starting retrieving cash_no_CIL composition..")
        self.retrieve_instrument_details()
        self.logger.info(".... retrievied cash_no_CIL composition!")
        self.create_NAV_matrix()

    def retrieve_pcf_composition(self):
        """
        Retrieves PCF NAVs from the database.
        """
        if self._date is not None:
            self.pcf_composition = self.pcf_manager.get_etf_composition(self._date, self._date, self.etf_list,
                                                                        use_old_names=True)
        else:
            self.pcf_composition = self.pcf_manager.get_last_etf_composition(self.etf_list, use_old_names=True)
        self.discard_small_components()
        self.isin_components = self.pcf_composition['BSH_ID_COMP'].unique().tolist()
        self.discard_ETFS_without_components()
        self.etf_to_ref_date = (self.pcf_composition[["BSH_ID_ETF", "REF_DATE"]].
                                drop_duplicates().
                                set_index("BSH_ID_ETF").to_dict())

    def create_physical_components_matrix(self, field="N_INSTRUMENTS"):
        """
        Creates a NAV matrix from the PCF composition.
        """
        self.physical_components_matrix = self.pcf_composition.pivot(index="BSH_ID_ETF",
                                                                     columns="BSH_ID_COMP",
                                                                     values=field).fillna(0)
        return self.physical_components_matrix

    def retrieve_instrument_details(self):
        """
        Retrieves instrument details from the database.
        """
        self.instrument_details = self.pcf_manager.get_instrument_details(self.isin_components +
                                                                          self.etf_list).set_index("BSH_ID")
        self.pcf_composition = self.pcf_composition.merge(self.instrument_details, left_on="BSH_ID_COMP",
                                                          right_on="BSH_ID", how="left")

    def create_cash_components_matrix(self, field="QUANTITY"):
        """
        Retrieves cash_no_CIL components from the database for each currency.
        """
        if self._date is not None:
            self.cash_composition = self.pcf_manager.get_etf_cash_components(self._date, self._date, self.etf_list,
                                                                             use_old_names=True)
        else:
            self.cash_composition = self.pcf_manager.get_last_etf_cash_components(self.etf_list, use_old_names=True)

        self.cash_composition = self.cash_composition.loc[self.cash_composition["CASH_IN_LIEU"] == "N"]
        self.cash_components_matrix = self.cash_composition.pivot_table(index='BSH_ID',
                                                                        columns='CURRENCY',
                                                                        values=field,
                                                                        aggfunc='sum').fillna(0)

        ref_date_check = (self.pcf_composition[["BSH_ID_ETF", "REF_DATE"]].
                          drop_duplicates().
                          set_index("BSH_ID_ETF").to_dict())["REF_DATE"] = \
            (self.cash_composition[["BSH_ID", "REF_DATE"]].
             drop_duplicates().
             set_index("BSH_ID").to_dict())["REF_DATE"]

        if not ref_date_check:
            raise ValueError("REF DATE are not equal for cash_no_CIL and pcf")
        
        self.process_fx_fwd_components()

    def create_NAV_matrix(self):
        """
        Creating and merging cash_no_CIL and physical components matrix
        :return:
        """

        self.create_physical_components_matrix()
        self.create_cash_components_matrix()
        self.nav_matrix = pd.merge(self.physical_components_matrix, self.cash_components_matrix,
                                   how='left',
                                   left_index=True,
                                   right_index=True)
        self.check_weight()
        self.nav_matrix.fillna(0, inplace=True)
        self.perform_ad_hoc_adjusting()

    def process_fx_fwd_components(self):
        """
        Processes FX forward components in the _securities.\
        """
        for security in self.pcf_composition[
            "BSH_ID_COMP"].unique():  # Use list to avoid modifying iterable during loop
            type = self.isin_classifier.classify(security)
            if type is Instrument.FXFWD:
                try:
                    fx = self.instrument_details.loc[security, "CURRENCY"]
                except KeyError as e:
                    fx = security[-3:]
                    print(e)
                self.process_single_fx_fwd_component(security, fx)
            elif type is Instrument.FUTURE:
                self.process_single_future_components(security)

    def get_weight_sum(self):

        return self.weight_sum

    def get_etf_to_ref_date(self):
        return pd.Series(self.etf_to_ref_date["REF_DATE"])

    def process_single_fx_fwd_component(self, security: str, fx: str):
        """
        Processes a single FX forward component.

        :param security: Security identifier.
        :param fx: FX identifier.
        """

        if fx in self.cash_components_matrix.columns:
            self.cash_components_matrix[fx] += self.physical_components_matrix[security]
        else:
            self.cash_components_matrix[fx] = self.physical_components_matrix[security]
        self.isin_components.remove(security)
        self.physical_components_matrix.drop(security, axis=1, inplace=True)

    def process_single_future_components(self, future: str):

        try:
            self.physical_components_matrix[future] *= download_future_multiplier(future)
        except Exception as e:
            self.logger.error(f"Exception in multiplying {future} for price mult: {e}")

    def discard_ETFS_without_components(self):
        """
        Discards missing PCF data.
        """
        if not hasattr(self, "pcf_composition"):
            self.logger.warning("Composition never fetched. Cannot obtain missing ETFs")
            return
        self.missing_pcfs = set(self.etf_list) - set(self.pcf_composition["BSH_ID_ETF"].values)
        missing_pcfs = [ISIN_TO_TICKER.get(etf) for etf in self.missing_pcfs]
        self.logger.warning("\n\n" + f"missing pcfs ({len(missing_pcfs)}): \n\n " + "\n ".join(missing_pcfs) + "\n")
        self.etf_list = [isin for isin in self.etf_list if isin not in self.missing_pcfs]

    def discard_small_components(self, bp_threshold=1) -> None:

        check = (self.pcf_composition["WEIGHT_NAV"].abs() < bp_threshold * 1e-5)
        if check.any():
            # if input("Want to drop small components?(Y/N): \n").upper() == "Y":
                self.logger.warning(f"Discarding small components\n\n")
                self.logger.info(self.pcf_composition.loc[check, ["BSH_ID_ETF",
                                                                     "BSH_ID_COMP",
                                                                     "WEIGHT_NAV"]].to_string(index=False) + "\n")
                self.pcf_composition = self.pcf_composition[~check]

    def check_weight(self):
        """Checks and prints the weight NAV for components."""
        self.logger.info("Checking weights ETFs components:")

        # Pivot tables for physical components and FX
        weight_check_matrix_physical = self.pcf_composition.pivot_table(
            index="BSH_ID_ETF", columns="BSH_ID_COMP", values="WEIGHT_RISK", aggfunc="sum"
        ).fillna(0)
        weight_check_fx = self.cash_composition.pivot_table(
            index="BSH_ID", columns="CURRENCY", values="WEIGHT_RISK", aggfunc="sum"
        ).fillna(0)

        self.weight_nav_matrix = pd.concat([weight_check_matrix_physical, weight_check_fx], axis=1)
        # self._correct_weights_per_multiplier()

        self.weight_sum = self.weight_nav_matrix.sum(axis=1).rename("Weight NAV")

        # Check for weights with null ISIN
        weight_with_null_isin = self.weight_nav_matrix[
            [col for col in self.weight_nav_matrix.columns if col.startswith("OTHER")]].sum(axis=1)
        weight_with_null_isin_errors = weight_with_null_isin[weight_with_null_isin != 0]

        if not weight_with_null_isin_errors.empty:
            self.logger.error(f"Weights with null ISIN are non-zero:\n{weight_with_null_isin_errors.to_string()}")

        self.weight_sum.sort_values(inplace=True)

        if len(etf_to_drop := self.weight_sum[self.weight_sum < 0.5]):
            self.logger.error(f"\nweight sum less than 50%\n" + etf_to_drop.to_string() + "\n")

        if len(etf_to_warn := self.weight_sum[(1 - self.weight_sum).abs() > 1e-4]):
            self.logger.warning(f"\nweight is less than {100 - 1e-3}%\n" + etf_to_warn.to_string() + "\n")

    def get_nav_matrix(self, weight=False):
        """
        Retrieves the NAV matrix.
        :return: NAV matrix.
        """
        if weight:
            return self.weight_nav_matrix
        else:
            return self.nav_matrix

    def get_securities(self):
        self.securities = list(set(self.nav_matrix.index.tolist() + self.nav_matrix.columns.tolist()))
        return [s for s in self.securities if s not in ["GBp", "OTHEREQUIEUR"]]

    def get_issuer_prices(self):
        if hasattr(self, "issuer_prices") and self.issuer_prices is not None:
            return self.issuer_prices
        else:
            issuer_prices = self.pcf_composition[["REF_DATE", "BSH_ID_ETF", "BSH_ID_COMP", "PRICE_FUND_CCY"]]
            if issuer_prices.isna().any().any():
                self.logger.info("\nIssuer price are none for these ETFs:\n"
                                 + "\n".join(issuer_prices.loc[issuer_prices.isna().any(axis=1)]["BSH_ID_ETF"].unique()))
                issuer_prices = issuer_prices[~issuer_prices.isna()]
            self.issuer_prices = issuer_prices
            return self.issuer_prices

    def get_etf_nav_ccy(self, isin: list) -> pd.DataFrame:
        start = end = dt.date.today() - BDay(3)
        return self.pcf_manager.get_etf_fund_nav_ccy(start, end, isin)

    def get_missing_pcfs(self):
        return self.missing_pcfs

    def get_components(self):
        cols = self.nav_matrix.columns
        return [c for c in cols if not c.startswith("OTHER")]

    def perform_ad_hoc_adjusting(self):
        pass


    def _correct_weights_per_multiplier(self):
        for instr in self.weight_nav_matrix.columns:
            if self.isin_classifier.is_future(instr):
                multiplier = download_future_multiplier(instr)
                self.weight_nav_matrix[instr] *= multiplier

@memoryPCF.cache
def download_future_multiplier(future: str):
    pass
    # flds = "price_multiplier"
    # mult = bdp(future, flds).loc[future, flds]
    # logging.info(f"multiplier of {future}: {mult}")
    # return mult

if __name__ == "__main__":
    isins = ISINS_ETF_EQUITY
    pc = PCFProcessor(isins)
    available_isins = pc.pcf_composition["BSH_ID_ETF"].unique()
    with open("available_isins.txt", "w") as file:
        file.write(str(available_isins))