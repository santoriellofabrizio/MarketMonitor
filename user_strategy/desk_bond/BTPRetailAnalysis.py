import datetime as dt
from functools import lru_cache
from typing import Dict, List, Tuple
import logging
import time
import pandas as pd
import xlwings as xw
import os

from sfm_quantlib.Builder.BuildConfig import BuildConfig
from sfm_quantlib.Dates.Calendar.CalendarFactory import CalendarFactory
from sfm_quantlib.Dates.Calendars import Calendars
from sfm_quantlib.Dates.FrequencyType import FrequencyType
from sfm_quantlib.Dates.TimeUnit import TimeUnit
from sfm_quantlib.FinancialUtilities.BusinessDayConvention import BusinessDayConvention
from sfm_quantlib.Instruments.Bonds.Bond import Bond
from sfm_quantlib.Instruments.Bonds.BondPricingModelType import BondPricingModelType
from sfm_quantlib.Instruments.Bonds.FixedRateBondsAnalytics import compute_ytm
from sfm_quantlib.Pricers.Model import Model
from sfm_quantlib.InstrumentsConnections.InstrumentFactory import InstrumentFactory

from market_monitor.publishers.redis_publisher import RedisPublisher, RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI


class BTPRetailAnalysis(StrategyUI):

    __mapping_side: Dict[int, str] = {1: "BUY", -1: "SELL"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        calendar = CalendarFactory().get_calendar(Calendars.TARGET)
        self._settlement_date: dt.datetime.date = calendar.advance_time_unit(
            dt.datetime.today().date(), 2, TimeUnit.DAYS, BusinessDayConvention.FOLLOWING)

        self._mids: Dict[str, float] = {}
        self.fields = ["MID"]

        # definisco le variabili lette dall'excel e le inizializzo attraverso la funzione
        self._workbook_name: str = kwargs['workbook_name']

        self._bonds_mapping: Dict[str, str] = {}
        self._retails: List[str] = []
        self._btps: List[str] = []
        self._futures: List[str] = []

        self._retails_description: Dict[str, str] = {}
        self._retails_hr: Dict[str, Tuple[float, float]] = {}
        self._historic_prices: Dict[str, float] = {}
        self._morning_basis: Dict[str, float] = {}

        self.__initialize_excel_variables()

        self._bbg_instruments: List[str] = self._btps + self._futures

        # Definisco le variabili per costruire i bonds
        self._db_enviroment: str = kwargs.get("db_enviroment")
        self._db_user: str = kwargs.get("db_user")
        self._db_password: str = kwargs.get("db_password")

        self._bonds: Dict[str, Bond] = {}
        self.__build_bond()

        self.redis_publisher = RedisMessaging()

        # definisco il df dove salvo i dati
        self._columns = ["ISIN", "DESCRIPTION", "MARKET", "TIME", "QTY", "PRICE", "SIDE", "DELTA YTM", "BASE"]
        self.trades: pd.DataFrame = pd.DataFrame(columns=self._columns)

    def on_market_data_setting(self) -> None:
        """
        Set the market data to include _securities and currency pairs.
        default for ccy is EURCCY. es EURUSD.
        """
        subscriptions: Dict[str, str] = {sec: f"{sec} @MILA Govt" for sec in self._btps}
        subscriptions[self._futures[0]] = f"{self._futures[0]} Comdty"
        subscriptions[self._futures[1]] = f"{self._futures[1]} Comdty"

        self.market_data.set_securities(self._bbg_instruments)
        for sec, subs in subscriptions.items():
            self.market_data.subscribe(id=sec, subscription_string=subs,fields=self.fields)

    def on_my_trade(self, trades: pd.DataFrame):
        """
        Processes trades by calculating delta YTM for each trade
        and forwarding the enriched DataFrame to the trade manager.

        Args:
            trades (pd.DataFrame): DataFrame with at least columns 'isin' and 'price'.
        """

        analyzed_trades = [self._analyze_trade(trade) for _, trade in trades.iterrows()
                           if trade['isin'] in self._retails]
        df = pd.DataFrame(analyzed_trades, columns=self._columns)

        self.trades = pd.concat([df, self.trades])
        self.trades.sort_values(by="TIME", inplace=True, ascending=False)

        self.redis_publisher.export_message(channel="desk_bond_trades",
                                            value=self.trades)

    def update_HF(self):
        self._mids = self.market_data.get_data_field(field=self.fields,
                                                     securities=self._bbg_instruments).to_dict()

    def _analyze_trade(self, trade: pd.Series) -> List:
        isin = trade['isin']
        description = self._retails_description[isin]
        delta_ytm = self._compute_delta(trade)

        hr1, hr2 = self._retails_hr[isin]
        fut1, fut2 = self._mids[self._futures[0]], self._mids[self._futures[1]]
        theoretical_price = fut1 * hr1 + fut2 * hr2 + self._morning_basis[isin]

        base = (trade['price'] - theoretical_price) * 100

        analyzed_trade = [trade['isin'], description, trade['market'], trade['last_update'], trade['quantity'],
                          trade['price'], self.__mapping_side[trade['own_trade']], delta_ytm, base]

        return analyzed_trade

    def _compute_delta(self, row):
        reference_isin = self._bonds_mapping.get(row["isin"])
        reference_price = self._mids[reference_isin]
        ytm_reference = self._compute_ytm(reference_isin, reference_price)
        ytm_quoting = self._compute_ytm(row["isin"], row["price"])
        return (ytm_quoting - ytm_reference) * 10_000

    def __build_bond(self):
        # builder = BuildConfig(self._quantlib_config_file)
        # configuration = builder.build_configuration()
        # reference_date = self._settlement_date
        # information_connection = configuration.information_connection
        # instruments = information_connection.get_instruments(
        #     {bond: 100_000 for bond in self._btps + self._retails}, reference_date)
        instrument_factory = InstrumentFactory().setup_db_information_connection(self._db_enviroment, self._db_user, self._db_password)
        instruments = instrument_factory.get_instruments({bond: 100_000 for bond in self._btps + self._retails})
        _ = [instr.set_model(Model.NO_MODEL) for instr in instruments]
        self._bonds = {bond.name: bond for bond in instruments}

    @lru_cache(maxsize=1000)
    def _compute_ytm(self, isin: str, price: float) -> float:
        try:
            ytm = compute_ytm(self._settlement_date, self._bonds[isin], price, BondPricingModelType.ISMA,
                              None, FrequencyType.ANNUAL)
            return ytm
        except Exception as e:
            print(f"Unable to compute ytm for {isin} and {price}")

        return 0

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

        self._bonds_mapping = {retail: btp for retail, btp in zip(
            bonds_df["ISIN"].values.tolist(), bonds_df["INSTRUMENT"].values.tolist())}
        self._retails: List[str] = list(self._bonds_mapping.keys())
        self._btps: List[str] = list(set(self._bonds_mapping.values()))

        self._retails_description = {retail: btp for retail, btp in zip(
            self._retails, bonds_df["Name"].values.tolist())}

        self._retails_hr: Dict[str, Tuple[float, float]] = {retail: (fbts, fbtp) for retail, fbts, fbtp in zip(
            self._retails, bonds_df["FBTS"].values.tolist(), bonds_df["FBTP"].values.tolist())}

        self._historic_prices: Dict[str, float] = {retail: yesterday_price[0] for retail, yesterday_price in zip(
            self._retails, bonds_df["Yesterday Price"].values.tolist())}

        futures_data = futures_sheet.range("A3").expand().value
        futures_df = pd.DataFrame(futures_data[1:], columns=futures_data[0])

        self._futures: List[str] = futures_df["Ticker"].values.tolist()[:2]
        self._historic_prices.update({fut: price for fut, price in zip(
            self._futures, futures_df["YESTERDAY PRICE"].values.tolist()[:2])})

        self.__initialize_basis()

    def __initialize_basis(self):
        fut1, fut2 = self._historic_prices[self._futures[0]], self._historic_prices[self._futures[1]]
        for bond in self._retails:
            hr1, hr2 = self._retails_hr[bond]
            self._morning_basis[bond] = self._historic_prices[bond] - hr1 * fut1 - hr2 * fut2
