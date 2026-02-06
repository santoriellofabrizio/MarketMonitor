import logging
from time import time, sleep
from datetime import datetime
import datetime as dt

import pandas as pd
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.equity.utils.SQLUtils.storage import PriceDatabaseManager
from user_strategy.utils import CustomBDay
from market_monitor.strategy.common.trade_manager.book_memory import BookStorage
from market_monitor.strategy.common.trade_manager.flow_detector import FlowDetector
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager


class EtfEquityLiveAnalysis(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.min_ctv_trades_excel = kwargs.get("min_ctv_trades_excel", 20000)
        self.price_source = kwargs.get("price_source", "kafka")

        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\MarketMonitor\etc\config\bshdata_config.yaml")

        self.all_isin_ETFP = self.API.general.get(fields=["etp_isins"],
                                                  segments=["IM"],
                                                  currency="EUR",
                                                  underlying="EQUITY",
                                                  source="oracle")["etp_isins"]

        self.isin_to_ticker = self.API.info.get_etp_fields(isin=self.all_isin_ETFP[:1000], fields="TICKER")[
            "TICKER"].to_dict()
        self.isin_to_ticker.update(
            self.API.info.get_etp_fields(isin=self.all_isin_ETFP[1000:], fields="TICKER")["TICKER"].to_dict())
        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}

        self.subscription_manager: None | SubscriptionManager = None
        self.book_storage: BookStorage = BookStorage()

        self.flow_detector = FlowDetector()

        self.yesterday = datetime.today() - CustomBDay

        self.instruments = self.all_isin_ETFP
        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.theoretical_price: pd.Series = pd.Series(index=self.instruments, name="TH_CLUSTER_PRICE")
        self.mid = pd.Series(index=self.all_isin_ETFP, name="mid")

        self.trade_dashboard_messaging = RedisMessaging()
        self.trade_manager = TradeManager(self.book_storage,
                                          self.theoretical_price,
                                          **kwargs["trade_manager"])

        self.price_db_manager = PriceDatabaseManager(db_path=
                                                     r"C:\Users\gbs09316\Desktop\etf_equity_historical_price.sqlite")

        self.last_storage_time = 0

    def wait_for_book_initialization(self):

        while datetime.today().time() < dt.time(9, 5):
            return False

        while self.price_source == "bloomberg" and not self.wait_for_bloomberg_initialization():
            sleep(1)

        self.on_start_strategy()
        return True

    def wait_for_bloomberg_initialization(self):

        while self.market_data.get_pending_subscriptions("bloomberg"):
            sleep(1)

        subscription_manager = self.market_data.get_subscription_manager()
        for sub in self.market_data.get_subscription_manager().get_failed_subscriptions():
            isin = sub.get("id")
            if isin in self.all_isin_ETFP:
                ticker = self.isin_to_ticker.get(isin)
                subscription_manager.subscribe_bloomberg(isin, f"{ticker} IM EQUITY", ["BID", "ASK"])

        return True

    def on_start_strategy(self):

        last_trades = self.trade_manager.get_trades()
        if not last_trades.empty:
            # last_trades["cluster"] = last_trades["isin"].map(self.clusters)
            self.publish_trades_on_dashboard(last_trades)

            self.trade_dashboard_messaging.export_message(channel="trades_df",
                                                          value=last_trades,
                                                          date_format='iso',
                                                          orient="records")

    def on_market_data_setting(self) -> None:
        # Subscribe to original channel names with market: prefix
        subscription_manager = self.market_data.get_subscription_manager()
        for channel in ["theoretical_live_intraday_price"]:
            subscription_manager.subscribe_redis(
                channel=f"market:{channel}",
                store="market"
            )
        for isin in self.all_isin_ETFP:
            if self.price_source == "bloomberg":
                subscription_manager.subscribe_bloomberg(isin, f"{isin} IM EQUITY", ["BID", "ASK"])
            else:
                subscription_manager.subscribe_kafka(id=isin,
                                                     symbol_filter=isin,
                                                     topic="COALESCENT_DUMA.ETFP.BookBest",
                                                     fields_mapping={
                                                         "BID": "bidBestLevel.price",
                                                         "ASK": "askBestLevel.price",
                                                         "BID_SIZE": "bidBestLevel.quantity",
                                                         "ASK_SIZE": "askBestLevel.quantity"})

        for isin in self.all_isin_ETFP:
            subscription_manager.subscribe_kafka(id=f"{isin}:PublicDeal",
                                                 symbol_filter=isin,
                                                 topic="COALESCENT_DUMA.ETFP.PublicDeal")

            subscription_manager.subscribe_kafka(id=f"{isin}:Trade",
                                                 symbol_filter=isin,
                                                 topic="COALESCENT_DUMA.ETFP.Trade")

    def update_LF(self) -> None:
        try:
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        if datetime.today().time() < dt.time(17, 29, 40):
            self.get_live_data()

        # payload = {
        #     'theoretical_intraday': self.theoretical_price,
        #     'mid': self.mid
        # }
        #
        # # Invio allo storage (la classe non sa cosa c'è dentro, si limita a processare)
        # if time() - self.last_storage_time > 10:
        #     self.price_db_manager.store_data(payload)
        #     self.last_storage_time = time()

    def on_trade(self, new_trades):

        processed_new = self.trade_manager.on_trade(new_trades)

        self.flow_detector.process_trades(processed_new)
        if self.flow_detector.has_new_flows():
            for flow in self.flow_detector.get_new_flows():  # ← Una volta sola!
                self.trade_dashboard_messaging.export_flow_detected(channel="trades_df", flow=flow)

        # Invia trades: nuovi parziali + parziali precedenti ora elaborati
        trades_to_publish = self.trade_manager.get_trades_to_publish(processed_new)
        # trades_to_publish["cluster"] = trades_to_publish[("isin"
        #                                                   "")].map(self.clusters)
        self.publish_trades_on_dashboard(trades_to_publish)
        self.publish_trades_to_excel(processed_new)

    def publish_trades_to_excel(self, processed_new):

        trades = self.trade_manager.get_trades(n_seconds=40)

        trades = trades.loc[(trades["own_trade"]) | (trades["ctv"] > self.min_ctv_trades_excel)]

        self.trade_dashboard_messaging.export_message(channel="trades_df_excel",
                                                      value=trades,
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

        self.theoretical_price.update(
            self.market_data.get_data_field(field="theoretical_live_intraday_price"))
        self.mid.update(self.market_data.get_data_field(field=["BID", "ASK"]).mean(axis=1))

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
