import time
from datetime import datetime

import numpy as np
import pandas as pd

from market_monitor.data_storage.NAVDataStorage import NAVDataStorage

from market_monitor.gui.implementations.GUI import GUI
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.equity.utils.DataProcessors.PCFControls import PCFControls
from user_strategy.equity.utils.DataProcessors.PCFProcessor import PCFProcessor
from user_strategy.equity.utils.DataProcessors.StockSelector import StockSelector
from user_strategy import InputParamsQuoting

from user_strategy.utils import CustomBDay
from user_strategy.utils.enums import ISIN_TO_TICKER, CURRENCY


class NavChecking(StrategyUI):

    def __init__(self, instruments, **kwargs) -> None:
        super().__init__(**kwargs)

        self.mid_eur: pd.Series | None = None
        self.fx_list: list | None = None
        self.securities_list: list | None = None
        self.gui: GUI
        self.storage = NAVDataStorage(db_name=kwargs.get("db_name", "data_storage/data_storage.db"))
        self.yesterday = datetime.today() - CustomBDay
        self.pcf_processor = PCFProcessor(etf_list=instruments, date=self.yesterday)
        self.pcf_controls = PCFControls(self.pcf_processor)
        self.nav_matrix = self.pcf_processor.get_nav_matrix()
        self.weight_nav_matrix = self.pcf_processor.get_nav_matrix(weight=True)
        self.stock_selector = StockSelector(self.weight_nav_matrix)

        stock_to_drop = self.stock_selector.get_stock_to_drop()
        self.nav_matrix.drop(stock_to_drop, axis=1, inplace=True, errors='ignore')
        self.weight_nav_matrix.drop(stock_to_drop, axis=1, inplace=True, errors='ignore')

        fx_correction_issuer = self.pcf_controls.convert_fund_ccy_to_eur()
        self.issuer_prices = self.pcf_controls.issuer_prices_data
        self.composition = self.pcf_processor.pcf_composition
        self.composition["PRICE_EUR"] = (self.composition["PRICE_FUND_CCY"] /
                                         self.composition["BSH_ID_ETF"].map(fx_correction_issuer))

        self.weight_correction = self.weight_nav_matrix.sum(axis=1)
        self._instruments = self.nav_matrix.index.tolist()
        self.issuer_prices = self.pcf_controls.get_issuer_prices()
        self.isins_components = list(set(self.pcf_processor.get_components()) - set(stock_to_drop))
        self.securities_list = list(set(self.pcf_processor.get_securities()) - set(stock_to_drop))
        self.isin_to_ticker = ISIN_TO_TICKER
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        live_params = {"isin_to_check": self.show_etf_to_check}
        self.input_params = InputParamsQuoting(live_params)
        self.output = pd.DataFrame(index=self.instruments, columns=["PRICE", "NAV"])
        self.output.index.name = "ETF"

        # self.instruments_status = self.market_data.instruments_status

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        """
        self.market_data.securities = list(set(self.securities_list) | CURRENCY)
        for isin, price in self.issuer_prices.items():
            if np.isnan(price) or price is None: continue
            if isin in self.market_data.securities:
                self.market_data.update(isin, {flds: price for flds in self.market_data.fields})

    def update_LF(self) -> None:

        if self.mid_eur is None: return

    def update_HF(self):

        self.mid_eur = self.market_data.get_mid_eur()
        self.instruments = [i for i in self.instruments if i in self.weight_correction.index]
        if self.mid_eur is None: return
        self.output["PRICE"] = self.mid_eur[self.instruments]
        self.output["NAV"] = (self.nav_matrix[self.isins_components] @
                              self.mid_eur[self.isins_components])

        self.output["NAV"].replace({0: np.nan}, inplace=True)
        self.output["WEIGHT"] = self.weight_correction[self.instruments]
        self.output["NAV"] /= self.output["WEIGHT"]

        return {"data": (self.output.rename(self.isin_to_ticker, axis='index').
                                    sort_values(by=["NAV"], ascending=False)),
                "cell": "A1",
                "sheet": "NavDashboard"}

    def wait_for_book_initialization(self):
        """
        Attende l'inizializzazione del book e gestisce strumenti con dati mancanti.
        """
        self.logger.info("Waiting for book initialization")
        time.sleep(3)
        mid = self.market_data.get_mid_eur()
        missing = mid[mid.isna()]
        for instr in missing.index:
            if input(f"\nSubscription of {instr}: Want to impute a value? Y/N ").strip().upper() == "Y":
                while True:
                    try:
                        price = float(input(f"Enter a price for {instr}: "))
                        self.market_data.update(instr,
                                                {fld: price for fld in self.market_data.get_available_fields("market")})
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
            else:
                self.market_data.update(instr, {fld: 0 for fld in self.market_data.get_available_fields("market")})
        return True

    def on_book_initialized(self):

        self.check_delisting_and_issuer_price()
        self.check_live_weights()
        self.check_live_and_issuer_price_diff()

    def store_data_on_DB(self):

        storage = self.output[["PRICE", "NAV"]].rename(self.isin_to_ticker, axis='index').reset_index()
        storage.round({"PRICE": 3, "NAV": 3})
        storage["DATETIME"] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        return {"NAV_PRICE": storage}

    def show_etf_to_check(self, instrument_to_check):
        instrument_to_check = self.as_isin(instrument_to_check)
        book = self.market_data.get_mid_eur()
        for etf in instrument_to_check:
            try:
                self._build_single_etf_report(etf, book)
            except Exception as e:
                self.logger.error(f"Error in generating report for {etf}:" + e)

    def _build_single_etf_report(self, isin: str, book: pd.DataFrame | pd.Series):
        isin = isin.strip().upper()
        if isin in self.instruments:
            cash = self.pcf_processor.cash_composition.loc[self.pcf_processor.cash_composition["BSH_ID"] == isin]
            comp = self.composition
            report = comp.loc[comp["BSH_ID_ETF"] == isin,
            ["BSH_ID_COMP", "N_INSTRUMENTS", "PRICE_EUR", "PRICE_FUND_CCY", "WEIGHT_RISK"]].set_index(
                "BSH_ID_COMP")
            report["MY PRICE EUR"] = [book[i] if i in book else np.nan for i in report.index]
            report["PRICE DIFF (ABS)"] = np.abs(report["MY PRICE EUR"] / report["PRICE_FUND_CCY"] - 1)
            for c in cash["CURRENCY"].unique(): report.loc[c] = 0

            for _, val in cash.loc[cash["BSH_ID"] == isin].iterrows():
                ccy = val["CURRENCY"]
                report.at[ccy, "N_INSTRUMENTS"] = val["QUANTITY"]
                report.at[ccy, "WEIGHT_RISK"] = val["WEIGHT_RISK"]
                report.at[ccy, "MY PRICE EUR"] = 1 if ccy == "EUR" else book[ccy]

            report["MY WEIGHT"] = report["N_INSTRUMENTS"] * report["MY PRICE EUR"] / self.output.loc[isin, "NAV"]
            report["WEIGHT DIFFERENCE"] = report["MY WEIGHT"] - report["WEIGHT_RISK"]
            report.sort_values(by="PRICE DIFF (ABS)", ascending=False, inplace=True)

            self.export_data({
                "data": report,
                "cell": "A1",
                "sheet": self.isin_to_ticker.get(isin, isin),
                "force": True})

        else:
            self.logger.warning(f"{isin} not present. Wrong spelling?")

    def check_live_weights(self):
        weights = self.pcf_processor.get_nav_matrix(weight=True)
        active_securities = self.market_data.get_securities_for_status("ACTV")
        self.live_weights = weights[[s for s in set(active_securities) if s in weights.columns]].sum(axis=1)
        error_live_weights = self.live_weights[self.live_weights < 0.90].sort_values(ascending=False)
        if len(error_live_weights): self.logger.warning(f"\nlow live weights for:\n\n" + error_live_weights.to_string())

    def check_live_and_issuer_price_diff(self):
        self.pcf_controls.check_for_issuers_price_errors()
        prices_check = self.pcf_controls.check_for_my_price_errors(self.market_data.get_mid_eur())
        self.export_data({"data": prices_check,
                          "cell": "A1",
                          "sheet": "MY PRICE CHECK"})

    def check_delisting_and_issuer_price(self):
        self.instruments_status = self.market_data.get_securities_for_status()
        for instr, status in self.instruments_status.items():
            if status != "ACTV":
                if instr in self.issuer_prices.index:
                    if isinstance(self.issuer_prices[instr], (pd.DataFrame, pd.Series)):
                        price = self.issuer_prices[instr].mean()
                    else:
                        price = self.issuer_prices[instr]
                    self.logger.warning(f"Instrument {instr} seems UNACTIVE,"
                                        f" but issuer price is {price:.3f}")

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = [self.as_isin(id) for id in value]

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments

    def as_isin(self, _id: str | list[str]) -> list[str] | str:
        if isinstance(_id, str): return self.ticker_to_isin.get(_id, _id)
        return [self.ticker_to_isin.get(el, el) for el in _id]
