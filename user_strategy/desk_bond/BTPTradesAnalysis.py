import datetime as dt
from collections import deque
from typing import Dict, List, Tuple
import logging
import time
import pandas as pd
import xlwings as xw
import os

from sfm_quantlib.Dates.Calendar.CalendarFactory import CalendarFactory
from sfm_quantlib.Dates.Calendars import Calendars
from sfm_quantlib.Dates.TimeUnit import TimeUnit
from sfm_quantlib.FinancialUtilities.BusinessDayConvention import BusinessDayConvention
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI


class BTPTradesAnalysis(StrategyUI):

    __mapping_side: Dict[int, str] = {1: "BUY", -1: "SELL"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        calendar = CalendarFactory().get_calendar(Calendars.TARGET)
        self._settlement_date: dt.datetime.date = calendar.advance_time_unit(
            dt.datetime.today().date(), 2, TimeUnit.DAYS, BusinessDayConvention.FOLLOWING)

        self._mids: Dict[str, float] = {}
        self.book_storage = deque([], maxlen=5)

        # definisco le variabili lette dall'excel e le inizializzo attraverso la funzione
        self._btps: List[str] = []
        self._futures: List[str] = []

        self._btps_description: Dict[str, str] = {}
        self._btps_hr: Dict[str, Tuple[float, float]] = {}
        self._historic_prices: Dict[str, float] = {}
        self._morning_basis: Dict[str, float] = {}

        self.__initialize_excel_variables()

        self._bbg_instruments = ([f"{sec}_MOT" for sec in self._btps] + [f"{sec}_MTS" for sec in self._btps]
                                 + self._futures)

        # definisco il df dove salvo i dati
        self._columns = ["ISIN", "DESCRIPTION", "MARKET", "TIME", "QTY", "PRICE", "SIDE", "SPREAD PL MOT",
                         "SPREAD PL MTS", "BASE"]
        self.trades: pd.DataFrame = pd.DataFrame(columns=self._columns)

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        default for ccy is EURCCY. es EURUSD.
        """
        subscriptions: Dict[str, str] ={}
        subscriptions.update({f"{sec}_MOT": f"{sec} @MILA Govt" for sec in self._btps})
        subscriptions.update({f"{sec}_MTS": f"{sec} @MTS Govt" for sec in self._btps})
        subscriptions[self._futures[0]] = f"{self._futures[0]} Comdty"
        subscriptions[self._futures[1]] = f"{self._futures[1]} Comdty"

        self.market_data.securities = self._bbg_instruments
        self.market_data.subscription_dict_bloomberg = subscriptions

    def on_my_trade(self, trades: pd.DataFrame):
        """
        Processes trades by calculating delta YTM for each trade
        and forwarding the enriched DataFrame to the trade manager.

        Args:
            trades (pd.DataFrame): DataFrame with at least columns 'isin' and 'price'.
        """

        analyzed_trades = [self._analyze_trade(trade) for _, trade in trades.iterrows() if trade['isin'] in self._btps]
        df = pd.DataFrame(analyzed_trades, columns=self._columns)

        self.trades = pd.concat([df, self.trades])
        self.trades.sort_values(by="TIME", inplace=True, ascending=False)

        self.export_data("Excel", data=self.trades, cell="A1", sheet="TRADES", index=False)

    def update_HF(self):
        self._mids = self.market_data.get_mid(self._bbg_instruments).to_dict()
        self.book_storage.append(self._mids.copy())

    def get_old_mid(self):
        if len(self.book_storage):
            return self.book_storage[0]
        return {instr: 0 for instr in self._bbg_instruments}

    def _analyze_trade(self, trade: pd.Series) -> List:
        isin = trade['isin']
        description = self._btps_description[isin]
        price, qty, side = trade['price'], trade['quantity'], trade['own_trade']
        qty = qty * 1_000_000 if trade['market'] in ("MTSC", "BTAM", "BV") else qty
        mids = self.get_old_mid()
        hr1, hr2 = self._btps_hr[isin]
        fut1, fut2 = mids[self._futures[0]], mids[self._futures[1]]
        theoretical_price = fut1 * hr1 + fut2 * hr2 + self._morning_basis[isin]
        base = (trade['price'] - theoretical_price) * 100

        spread_pl_mot = qty * side * (mids[f"{isin}_MOT"] - price) * 0.01
        spread_pl_mts = qty * side * (mids[f"{isin}_MTS"] - price) * 0.01

        analyzed_trade = [trade['isin'], description, trade['market'], trade['last_update'], qty, price,
                          self.__mapping_side[side], spread_pl_mot, spread_pl_mts, base]

        return analyzed_trade

    def __initialize_excel_variables(self):
        my_wb: None | xw.Book = None
        max_count, count = 10, 0
        while count < max_count:
            try:
                for app in xw.apps:
                    for wb in app.books:
                        if os.path.basename(wb.name) == os.path.basename(self._workbook_name):
                            my_wb = wb
                break  # Breaks out of the loop if the file isn't found already open
            except Exception as e:
                logging.error(f"Errore nell'esaminare i workbook aperti: {e}")
                time.sleep(1)  # Attende un secondo prima di riprovare
                count += 1

        bonds_sheet = my_wb.sheets['Anagrafica Titoli']
        futures_sheet = my_wb.sheets['Anagrafica Future']

        bonds_data = bonds_sheet.range("A3").expand().value
        bonds_df = pd.DataFrame(bonds_data[1:], columns=bonds_data[0])

        self._btps: List[str] = bonds_df["ISIN"].values.tolist()

        self._btps_description = {btp: desc for btp, desc in zip(self._btps, bonds_df["Name"].values.tolist())}

        self._btps_hr: Dict[str, Tuple[float, float]] = {btp: (fbts, fbtp) for btp, fbts, fbtp in zip(
            self._btps, bonds_df["FBTS"].values.tolist(), bonds_df["FBTP"].values.tolist())}

        self._historic_prices: Dict[str, float] = {btp: yesterday_price for btp, yesterday_price in zip(
            self._btps, bonds_df["Yesterday Price"].values.tolist())}

        futures_data = futures_sheet.range("A3").expand().value
        futures_df = pd.DataFrame(futures_data[1:], columns=futures_data[0])

        self._futures: List[str] = futures_df["Ticker"].values.tolist()[:2]
        self._historic_prices.update({fut: price for fut, price in zip(
            self._futures, futures_df["YESTERDAY PRICE"].values.tolist()[:2])})

        self.__initialize_basis()

    def __initialize_basis(self):
        fut1, fut2 = self._historic_prices[self._futures[0]], self._historic_prices[self._futures[1]]
        for bond in self._btps:
            hr1, hr2 = self._btps_hr[bond]
            self._morning_basis[bond] = self._historic_prices[bond] - hr1 * fut1 - hr2 * fut2
