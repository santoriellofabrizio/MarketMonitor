import logging
from time import time, sleep
from collections import deque
from datetime import datetime
from typing import Type

import pandas as pd

from market_monitor.data_storage.NAVDataStorage import NAVDataStorage
from market_monitor.gui.implementations.GUI import GUI
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.ETFEquity.LiveQuoting.InputParamsQuoting import InputParamsQuoting

from user_strategy.ETFEquity.utils.DataProcessors.PCFControls import PCFControls
from user_strategy.ETFEquity.utils.DataProcessors.PCFProcessor import PCFProcessor
from user_strategy.ETFEquity.utils.DataProcessors.StockSelector import StockSelector
from user_strategy.utils import CustomBDay
from user_strategy.utils.trade_manager.flow_detector import FlowDetector
from user_strategy.utils.trade_manager.trade_manager import TradeManager
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager
from user_strategy.utils.enums import ISIN_TO_TICKER, CURRENCY

future_mapping = {"FESX202503": "IE00B53L3W79",  # CSSX5E
                  "FXXP202503": "LU0908500753",  # MEUD
                  "FDXM202503": "LU0274211480",  # CG1
                  "FESB202503": "LU1834983477"}  # BNK


class EtfEquityLiveAnalysis(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.isin_to_ticker = ISIN_TO_TICKER
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        self.ctv: pd.Series | None = None
        self.subscription_manager: None | SubscriptionManager = None
        self.activate_stock_exposure: bool = kwargs.get("activate_stock_exposure", True)
        self.activate_stock_risk_calculator: bool = kwargs.get("activate_stock_risk_calculator", True)
        self.theoretical_prices: pd.DataFrame | None = None
        self.book_storage: deque = deque(maxlen=3)
        self.spreads: pd.Series | None = None
        self.strategy_input: pd.DataFrame | pd.Series | None = None
        self.position: pd.Series | None = None
        self.drivers = [self.as_isin(d) for d in kwargs.get("drivers", [])]
        self.live_weights: pd.Series | None = None
        self.mid_eur: pd.Series | None = None
        self.fx_list: list | None = None
        self.securities_list: list | None = None
        self.instruments_status: None | pd.Series = None
        self.GUIs: Type[GUI]
        self.flow_detector = FlowDetector()
        self.quoting_isins = self.as_isin(kwargs.get("quoting_isins", []))

        self.input_params = InputParamsQuoting(**kwargs)
        self.instruments_cluster = self.input_params.isin_cluster
        self.instruments_NAV = self.input_params.isin_nav

        self.storage = NAVDataStorage(db_name=kwargs.get("db_name", "data_storage/data_storage.db"))

        self.input_sheet = kwargs.get("PositionInputSheet")
        self.yesterday = datetime.today() - CustomBDay
        self.nav_matrix = pd.DataFrame()
        if self.instruments_NAV:
            self.pcf_processor = PCFProcessor(etf_list=self.instruments_NAV, date=self.yesterday)
            self.missing_isins = self.pcf_processor.get_missing_pcfs()
            self.classifier = self.pcf_processor.isin_classifier
            self.pcf_controls = PCFControls(self.pcf_processor)
            self.nav_matrix = self.pcf_processor.get_nav_matrix()
            self.instruments_NAV = self.nav_matrix.index.tolist()
            self.weight_nav_matrix = self.pcf_processor.get_nav_matrix(weight=True)
            self.stock_exposure: pd.Series | None = None

            # ------------------------------------ STOCK COMPRESSION --------------------------------------------------

            self.stock_selector = StockSelector(self.weight_nav_matrix)
            stock_to_keep, etf_to_keep = self.stock_selector.truncate_stocks(2500)
            stock_to_keep += list(CURRENCY)
            stock_to_keep = [stock for stock in stock_to_keep if stock not in kwargs.get("stock_to_ignore", [])]
            self.nav_matrix = self.nav_matrix.loc[etf_to_keep, self.nav_matrix.columns.isin(stock_to_keep)]
            self.weight_nav_matrix = self.weight_nav_matrix.loc[etf_to_keep,
            self.weight_nav_matrix.columns.isin(stock_to_keep)]
            self.weight_correction = self.weight_nav_matrix.sum(axis=1)
            self.composition = self.pcf_processor.pcf_composition
            self.composition["PRICE_EUR"] = (self.composition["PRICE_FUND_CCY"] /
                                             self.composition["BSH_ID_ETF"].
                                             map(self.pcf_controls.convert_fund_ccy_to_eur()))

            self.issuer_prices = self.pcf_controls.get_issuer_prices()

        self.instruments = list(set(self.instruments_NAV + self.instruments_cluster))

        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.nav: pd.Series = pd.Series(index=self.instruments_NAV, name="NAV")
        self.theoretical_live_cluster_price: pd.Series = pd.Series(index=self.instruments, name="TH_CLUSTER_PRICE")
        self.mid = pd.Series(index=self.instruments, name="market_best")

        self.trade_manager = TradeManager(self.book_storage,
                                          self.theoretical_live_cluster_price,
                                          **kwargs["trade_manager"])

    def wait_for_book_initialization(self):
        while True:
            data = self.market_data.get_data_field(field="mid", index_data="market")
            if data is not None and not data.empty:
                break
            sleep(1)
        return True

    def on_market_data_setting(self) -> None:
        # Subscribe to original channel names with market: prefix
        subscription_manager = self.market_data.get_subscription_manager()
        for channel in ["nav", "mid", "theoretical_live_cluster_price"]:
            subscription_manager.subscribe_redis(
                channel=f"market:{channel}",
                store="market"
            )

    def update_LF(self) -> None:
        try:
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        self.get_live_data()

    def on_trade(self, new_trades):

        processed_new = self.trade_manager.on_trade(new_trades)

        # self.flow_detector.process_trades(processed_new)
        # if self.flow_detector.has_new_flows():
        #     for flow in self.flow_detector.get_new_flows():  # ← Una volta sola!
        #         self.GUIs["TradeDashboard"].export_flow_detected(channel="trades", flow=flow)

        last_trades = self.trade_manager.get_trades(len(processed_new) + 10)
        self.publish_trades_on_dashboard(last_trades)

        self.GUIs["TradeDashboard"].export_message(channel="trades_df",
                                                   value=last_trades.drop(["is_elaborated"],
                                                                          errors='ignore'),
                                                   date_format='iso',
                                                   orient="records")

    def publish_trades_on_dashboard(self, new_trades):

        start = time()
        self.GUIs["TradeDashboard"].export_message(channel="trades",
                                                   value=new_trades,
                                                   date_format='iso',
                                                   orient="records")

        logging.debug(f"processing trades takes {time() - start} seconds.")

    def on_book_initialized(self):
        pass

    def store_data_on_DB(self):

        prices = pd.concat([self.nav,
                            self.mid.rename("PRICE"),
                            self.theoretical_live_cluster_price], axis=1)
        prices.index.name = "ETF"
        time_snip = datetime.now()
        if time_snip < time_snip.replace(hour=9, minute=15) or time_snip > time_snip.replace(hour=17, minute=20):
            return None
        storage = prices.rename(self.isin_to_ticker, axis='index').reset_index()
        storage.round({"PRICE": 3,
                       "NAV": 3,
                       "TH_DRIVER_PRICE": 3,
                       "TH_CLUSTER_PRICE": 3})

        storage["DATETIME"] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        return {"NAV_PRICE": storage}

    def get_live_data(self):
        # Read from MarketStore using original channel names
        self.nav.update(self.market_data.get_data_field(field="nav", index_data="market"))
        self.theoretical_live_cluster_price.update(
            self.market_data.get_data_field(field="theoretical_live_cluster_price", index_data="market"))
        self.mid.update(self.market_data.get_data_field(field="mid", index_data="market"))

        # Store book with timestamp
        self.book_storage.append((datetime.now(), self.mid.copy()))

    def on_stop(self):
        self.trade_manager.close()

    @property
    def instruments_NAV(self) -> list:
        return self._instruments_NAV

    @instruments_NAV.setter
    def instruments_NAV(self, value: list):
        self._instruments_NAV = [self.as_isin(id) for id in value]

    @instruments_NAV.getter
    def instruments_NAV(self) -> list[str]:
        return self._instruments_NAV

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = [self.as_isin(id) for id in set(value)]

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments

    def as_isin(self, _id: str | list[str]) -> list[str] | str:
        if isinstance(_id, str): return self.ticker_to_isin.get(_id, _id)
        return [self.ticker_to_isin.get(el, el) for el in _id]
