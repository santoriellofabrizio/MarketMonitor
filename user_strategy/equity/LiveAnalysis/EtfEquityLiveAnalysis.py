import logging
import sqlite3
from time import time, sleep
from datetime import datetime
import datetime as dt

import numpy as np
import pandas as pd
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.publishers.rabbit_publisher import RabbitMessaging
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

from market_monitor.strategy.common.trade_manager.book_memory import BookStorage, FairvaluePrice
from market_monitor.strategy.common.trade_manager.flow_detector import FlowDetector


class EtfEquityLiveAnalysis(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.quoting_instances = []
        self.price_source = kwargs.get("price_source", "kafka")

        self.API = BshData(config_path=r"C:\AFMachineLearning\Libraries\MarketMonitor\etc\config\bshdata_config.yaml")
        self.all_isins = self.API.general.get(fields=["etp_isins"],
                                              segments=["IM", "FP", "NA"],
                                              currency="EUR",
                                              underlying="EQUITY",
                                              source="oracle")["etp_isins"]

        leverage = self.API.info.get_etp_fields(isin=self.all_isins, source='bloomberg',
                                                fields="FUND_LEVERAGE")["FUND_LEVERAGE"].to_dict()

        etp_type = self.API.info.get_etp_fields(isin=self.all_isins,
                                                fields="ETP_TYPE")["ETP_TYPE"].to_dict()

        self.all_isins = [isin for isin in self.all_isins if
                          (leverage.get(isin, "N") == "N" and etp_type.get(isin, 'ETF') == 'ETF')]

        self.isin_to_ticker = self.API.info.get_etp_fields(isin=self.all_isins,
                                                           fields="TICKER")["TICKER"].to_dict()

        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}

        self.book_storage: BookStorage = BookStorage()
        self.flow_detector = FlowDetector()

        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.mid = pd.Series(index=self.all_isins, name="mid")
        self.model_price: pd.Series = pd.Series(np.nan, index=self.all_isins, name='model_price')

        self._init_redis_dashboard(**kwargs)
        self._init_rabbit_dashboard(**kwargs)

        self.trade_manager = TradeManager(self.book_storage, **kwargs.get("trade_manager", {}))
        self.quoting_instances = [instance for instance, _bool
                                  in self.redis_dashboard.get_key('quoting_instances').items() if _bool]
        self.isin_cluster_mapping: dict = self.get_clusters(path=kwargs.get('path_db'),
                                                            cluster_layer=kwargs.get('cluster_layer',
                                                                                     'FINAL_CLUSTER'))

    @staticmethod
    def get_clusters(path: str, cluster_layer: str = 'FINAL_CLUSTER') -> dict:
        if not path:
            return {}
        with sqlite3.connect(path) as conn:
            isin_cluster_mapping = pd.read_sql(
                f"""
                              SELECT isin, cluster_value
                              FROM isin_clusters
                              WHERE (isin, last_updated) IN (
                                  SELECT isin, MAX(last_updated)
                                  FROM isin_clusters
                                  GROUP BY isin
                              ) AND cluster_layer = '{cluster_layer}'
                              """, conn
            ).set_index('isin')['cluster_value'].to_dict()

        return isin_cluster_mapping

    def _init_rabbit_dashboard(self, **kwargs):
        rabbit_cfg = kwargs.get('rabbit_data_export', {})

        if rabbit_cfg.get('activate', False):
            rabbit_params = rabbit_cfg.get('rabbit_params', {})
            self.channel_rabbit = rabbit_params.get("channel_rabbit", "trades_rabbit")
            self.rabbit_dashboard = RabbitMessaging(**rabbit_params)
        else:
            self.rabbit_dashboard = None

    def _init_redis_dashboard(self, **kwargs):
        redis_cfg = kwargs.get('redis_data_export', {})
        if redis_cfg.get('activate', False):
            redis_params = redis_cfg.get('redis_params', {})
            self.channel_redis = redis_cfg.get("channel_redis", "trades_redis")
            self.redis_dashboard = RedisMessaging(**redis_params)
        else:
            self.redis_dashboard = None

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

        for sub in self.global_subscription_service.get_failed_subscriptions():
            isin = sub.get("id")
            if isin in self.all_isins:
                ticker = self.isin_to_ticker.get(isin)
                self.global_subscription_service.subscribe_bloomberg(isin, f"{ticker} IM EQUITY", ["BID", "ASK"])

        return True

    def on_start_strategy(self):
        last_trades = self.trade_manager.get_trades()
        if not last_trades.empty:
            self.publish_trades_on_dashboard(last_trades)

    def on_market_data_setting(self) -> None:

        for isin in self.all_isins:
            if self.price_source == "bloomberg":
                self.global_subscription_service.subscribe_bloomberg(isin, f"{isin} IM EQUITY", ["BID", "ASK"])
            else:
                self.global_subscription_service.subscribe_kafka(id=isin,
                                                                 symbol_filter=isin,
                                                                 topic="COALESCENT_DUMA.ETFP.BookBest",
                                                                 fields_mapping={
                                                                     "BID": "bidBestLevel.price",
                                                                     "ASK": "askBestLevel.price",
                                                                     "BID_SIZE": "bidBestLevel.quantity",
                                                                     "ASK_SIZE": "askBestLevel.quantity"})
        self.subscribe_kafka_trades()
        self.global_subscription_service.subscribe_redis(channel="market:theoretical_live_intraday_price",
                                                         store='market')

        self.quoting_instances = [instances for instances, _bool
                                  in self.redis_dashboard.get_key('quoting_instances').items() if _bool]

    def from_kafka_to_bloomberg(self):

        if self.price_source == 'kafka':
            for isin in self.all_isins:
                self.global_subscription_service.unsubscribe(isin, 'bloomberg')
                self.global_subscription_service.subscribe_bloomberg(isin, f"{isin} IM EQUITY",
                                                                     ["BID", "ASK"])
            self.price_source = 'bloomberg'

        elif self.price_source == 'bloomberg':
            logging.error(f"bloomberg is already the selected price source")
        else:
            logging.warning(f"Price source {self.price_source} not supported.")

    def from_bloomberg_to_kafka(self):
        if self.price_source == 'bloomberg':
            for isin in self.all_isins:
                self.global_subscription_service.unsubscribe(isin, 'bloomberg')
                self.global_subscription_service.subscribe_kafka(id=isin,
                                                                 symbol_filter=isin,
                                                                 topic="COALESCENT_DUMA.ETFP.BookBest",
                                                                 fields_mapping={
                                                                     "BID": "bidBestLevel.price",
                                                                     "ASK": "askBestLevel.price",
                                                                     "BID_SIZE": "bidBestLevel.quantity",
                                                                     "ASK_SIZE": "askBestLevel.quantity"})
            self.price_source = 'kafka'
        elif self.price_source == 'kafka':
            logging.error(f"kafka is already the selected price source")
        else:
            logging.warning(f"Price source {self.price_source} not supported.")

    def subscribe_kafka_trades(self) -> None:

        for isin in self.all_isins:
            for mkt in ["ETFP", "XPAR", "XAMS"]:
                self.global_subscription_service.subscribe_trades_kafka(id=f"{isin}:{mkt}:PublicDeal",
                                                                 symbol_filter=isin,
                                                                 topic=f"COALESCENT_DUMA.{mkt}.PublicDeal")

                self.global_subscription_service.subscribe_trades_kafka(id=f"{isin}:{mkt}:Trade",
                                                                 symbol_filter=isin,
                                                                 topic=f"COALESCENT_DUMA.{mkt}.Trade")

    def update_LF(self) -> None:
        try:
            # Drain progressive horizon updates first (thread-safe, non-blocking)
            horizon_updates = self.trade_manager.get_horizon_updates()
            if not horizon_updates.empty:
                self.publish_trades_on_dashboard(horizon_updates)
            # Full snapshot for consistency
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        if datetime.today().time() < dt.time(17, 29, 40):
            self.get_live_data()

    def on_trade(self, new_trades):

        new_trades = self._enrich_trades(new_trades)
        processed_new = self.trade_manager.on_trade(new_trades)

        self.flow_detector.process_trades(processed_new)
        if self.flow_detector.has_new_flows():
            for flow in self.flow_detector.get_new_flows():
                for dashboard in [self.rabbit_dashboard, self.rabbit_dashboard]:
                    if dashboard: dashboard.export_flow_detected(channel="trades_rabbit", flow=flow)

        trades_to_publish = self.trade_manager.get_trades_to_publish()
        self.publish_trades_on_dashboard(trades_to_publish)

    def _enrich_trades(self, trades) -> pd.DataFrame:
        trades['model_price'] = trades['isin'].map(self.model_price)
        trades['quoting'] = trades.apply(lambda row: f"{row.exchange}-{row.name}" in self.quoting_instances, axis=1)
        if self.isin_cluster_mapping:
            trades['cluster'] = trades['isin'].map(self.isin_cluster_mapping)
        return trades

    def publish_trades_on_excel(self):

        trades_to_publish = self.trade_manager.get_trades(n_seconds=10)
        if trades_to_publish.empty:
            pass

        trades_to_publish.drop([c for c in trades_to_publish.columns if "spread" in c],
                               inplace=True,
                               errors="ignore",
                               axis=1)

        trades_to_publish.drop(["is_elaborated",
                                "model_price",
                                "price_multiplier",
                                "description"], axis=1,
                               inplace=True)

        trades_to_publish["quoting"] = trades_to_publish.apply(lambda row:
                                                               f"{row.exchange}-{row.name}" in self.quoting_instances,
                                                               axis=1)

        self.redis_dashboard.export_message(channel=f"{self.channel_redis}_excel",
                                            value=trades_to_publish,
                                            date_format='iso',
                                            orient="records")

    def publish_trades_on_dashboard(self, new_trades):

        if self.redis_dashboard:
            self.redis_dashboard.export_message(channel=self.channel_redis,
                                                value=new_trades,
                                                date_format='iso',
                                                orient="records")
        if self.rabbit_dashboard:
            self.rabbit_dashboard.export_message(channel=self.channel_rabbit,
                                                 value=new_trades,
                                                 date_format='iso',
                                                 orient="records")

    def on_command(self, action: str, payload: dict):
        logging.warning('command arrived: {}'.format(action))
        if action == "switch_to_kafka":
            self.from_bloomberg_to_kafka()
        elif action == "switch_to_bloomberg":
            self.from_kafka_to_bloomberg()

    def get_live_data(self):
        raw = self.market_data.get_data_field(field=["BID", "ASK"])
        if raw is None or raw.empty:
            return

        model = self.market_data.get_data_field(field="theoretical_live_intraday_price")
        self.model_price.update(model)
        snapshot = {
            isin: FairvaluePrice.scalar(isin, price)
            for isin, price in raw.mean(axis=1).items()
        }
        self.book_storage.append(snapshot)

    def on_stop(self):
        self.trade_manager.close()
