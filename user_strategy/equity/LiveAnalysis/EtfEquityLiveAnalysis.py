import logging
from time import time, sleep
from collections import deque
from datetime import datetime
import datetime as dt

import numpy as np
import pandas as pd
from sfm_dataprovider.interface.bshdata import BshData

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


class EtfEquityLiveAnalysis(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\BshDataProvider\config\bshdata_config.yaml")

        self.all_isin_ETFP = self.API.general.get_etp_isins(underlying="EQUITY", segments="IM")
        self.clusters = {isin: np.random.choice(["EU", "BRASIL", "WORLD", "JAPAN", "CHINA"])
                         for isin in self.all_isin_ETFP}
        self.isin_to_ticker = self.API.info.get_etp_fields(isin=self.all_isin_ETFP[:1000], fields="TICKER")["TICKER"].to_dict()
        self.isin_to_ticker.update(self.API.info.get_etp_fields(isin=self.all_isin_ETFP[1000:], fields="TICKER")["TICKER"].to_dict())
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}

        self.subscription_manager: None | SubscriptionManager = None
        self.theoretical_prices: pd.DataFrame | None = None
        self.book_storage: BookStorage = BookStorage()

        self.mid_eur: pd.Series | None = None
        self.flow_detector = FlowDetector()

        self.yesterday = datetime.today() - CustomBDay

        self.instruments = self.all_isin_ETFP
        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.theoretical_live_cluster_price: pd.Series = pd.Series(index=self.instruments, name="TH_CLUSTER_PRICE")
        self.mid = pd.Series(index=self.all_isin_ETFP, name="mid")

        self.trade_dashboard_messaging = RedisMessaging()
        self.instruments = self.all_isin_ETFP
        self.trade_manager = TradeManager(self.book_storage,
                                          self.theoretical_live_cluster_price,
                                          **kwargs["trade_manager"])

    def wait_for_book_initialization(self):

        while datetime.today().time() < dt.time(9, 5):
            return False

        while True:
            data = self.market_data.get_data_field(field="MID")
            if data is not None and not data.empty:
                break
            sleep(1)
        self.on_start_strategy()
        return True

    def on_start_strategy(self):

        last_trades = self.trade_manager.get_trades()
        if not last_trades.empty:
            last_trades["cluster"] = last_trades["isin"].map(self.clusters)
            self.publish_trades_on_dashboard(last_trades)

            self.trade_dashboard_messaging.export_message(channel="trades_df",
                                                      value=last_trades,
                                                      date_format='iso',
                                                      orient="records")

    def on_market_data_setting(self) -> None:
        # Subscribe to original channel names with market: prefix
        subscription_manager = self.market_data.get_subscription_manager()
        for channel in ["nav", "theoretical_live_cluster_price"]:
            subscription_manager.subscribe_redis(
                channel=f"market:{channel}",
                store="market"
            )
        for isin in self.all_isin_ETFP:
            subscription_manager.subscribe_bloomberg(isin, f"{isin} IM EQUITY", ["MID"])

    def update_LF(self) -> None:
        try:
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        if datetime.today().time() < dt.time(17, 29, 40):
            self.get_live_data()
        self.get_live_data()

    def on_trade(self, new_trades):

        processed_new = self.trade_manager.on_trade(new_trades)

        self.flow_detector.process_trades(processed_new)
        if self.flow_detector.has_new_flows():
            for flow in self.flow_detector.get_new_flows():  # ← Una volta sola!
                self.trade_dashboard_messaging.export_flow_detected(channel="trades_df", flow=flow)

        # Invia trades: nuovi parziali + parziali precedenti ora elaborati
        trades_to_publish = self.trade_manager.get_trades_to_publish(processed_new)
        trades_to_publish["cluster"] = trades_to_publish[("isin"
                                                          "")].map(self.clusters)
        self.publish_trades_on_dashboard(trades_to_publish)

        self.trade_dashboard_messaging.export_message(channel="trades_df_excel",
                                                      value=self.trade_manager.get_trades(n_of_trades=20),
                                                      date_format='iso',
                                                      orient="records")

    def publish_trades_on_dashboard(self, new_trades):

        start = time()
        self.trade_dashboard_messaging.export_message(channel="trades_df",
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
        self.mid.update(self.market_data.get_data_field(field="MID"))

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
