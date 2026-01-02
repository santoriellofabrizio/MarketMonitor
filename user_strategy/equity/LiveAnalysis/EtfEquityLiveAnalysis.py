import logging
from time import time, sleep
from collections import deque
from datetime import datetime
from typing import Type

import pandas as pd

from market_monitor.gui.implementations.GUI import GUI
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.StrategyUI.StrategyUI import StrategyUI
from user_strategy.equity.LiveQuoting.InputParamsQuoting import InputParamsQuoting

from user_strategy.equity.utils.DataProcessors.PCFControls import PCFControls
from user_strategy.equity.utils.DataProcessors.PCFProcessor import PCFProcessor
from user_strategy.equity.utils.DataProcessors.StockSelector import StockSelector
from user_strategy.utils import CustomBDay
from user_strategy.utils.trade_manager.book_memory import BookStorage
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

        self.subscription_manager: None | SubscriptionManager = None
        self.theoretical_prices: pd.DataFrame | None = None
        self.book_storage: BookStorage = BookStorage()
        self.spreads: pd.Series | None = None

        self.mid_eur: pd.Series | None = None
        self.fx_list: list | None = None
        self.securities_list: list | None = None
        self.instruments_status: None | pd.Series = None
        self.flow_detector = FlowDetector()
        self.quoting_isins = self.as_isin(kwargs.get("quoting_isins", []))

        self.input_params = InputParamsQuoting(**kwargs)
        self.instruments_cluster = self.input_params.isin_cluster

        self.yesterday = datetime.today() - CustomBDay

        self.instruments = self.instruments_cluster

        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.theoretical_live_cluster_price: pd.Series = pd.Series(index=self.instruments, name="TH_CLUSTER_PRICE")
        self.mid = pd.Series(index=self.instruments, name="mid")

        self.trade_dashboard_messaging = RedisMessaging()

        self.trade_manager = TradeManager(self.book_storage,
                                          self.theoretical_live_cluster_price,
                                          **kwargs["trade_manager"])

    def wait_for_book_initialization(self):
        while True:
            data = self.market_data.get_data_field(field="mid")
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

        self.flow_detector.process_trades(processed_new)
        if self.flow_detector.has_new_flows():
            for flow in self.flow_detector.get_new_flows():  # ← Una volta sola!
                self.trade_dashboard_messaging.export_flow_detected(channel="trades", flow=flow)

        last_trades = self.trade_manager.get_trades(len(processed_new) + 10)
        self.publish_trades_on_dashboard(last_trades)

        self.trade_dashboard_messaging.export_message(channel="trades",
                                                      value=last_trades.drop(["is_elaborated"],
                                                                             errors='ignore'),
                                                      date_format='iso',
                                                      orient="records")

    def publish_trades_on_dashboard(self, new_trades):

        start = time()
        self.trade_dashboard_messaging.export_message(channel="trades",
                                                   value=new_trades,
                                                   date_format='iso',
                                                   orient="records")

        logging.debug(f"processing trades takes {time() - start} seconds.")

    def on_book_initialized(self):
        pass

    def get_live_data(self):
        # Read from MarketStore using original channel names
        self.theoretical_live_cluster_price.update(
            self.market_data.get_data_field(field="theoretical_live_cluster_price"))
        self.mid.update(self.market_data.get_data_field(field="mid"))

        # Store book with timestamp
        self.book_storage.append(self.mid.copy())

    def on_stop(self):
        self.trade_manager.close()

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
