import logging
from datetime import datetime
import datetime as dt
from typing import Dict, List

import pandas as pd

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.publishers.rabbit_publisher import RabbitMessaging
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

from market_monitor.strategy.common.trade_manager.book_memory import BookStorage, FairvaluePrice


class VTVFutAnalysis(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.all_isins = ["DE000F2Y2EW5", "DE000F2Y2EX3"]

        self.book_storage: BookStorage = BookStorage()

        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------

        self.mid = pd.Series(index=self.all_isins, name="mid")

        self._init_redis_dashboard(**kwargs)
        self._init_rabbit_dashboard(**kwargs)

        self._trade_manager_dict: Dict[int, TradeManager] = {}
        time_zero_lag_list = kwargs["trade_manager"]["time_zero_lag"]
        for time_zero_lag in time_zero_lag_list:
            time_zero_lag = int(time_zero_lag)
            self._trade_manager_dict[time_zero_lag] = TradeManager(self.book_storage, time_zero_lag=time_zero_lag)

        self._columns_to_filter: List[str] = []

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

        self.on_start_strategy()
        return True


    def on_start_strategy(self):
        pass


    def on_market_data_setting(self) -> None:

        for isin in self.all_isins:
            self.global_subscription_service.subscribe_kafka(id=isin,
                                                                 symbol_filter=isin,
                                                                 topic="COALESCENT_DUMA.XEUR.BookBest",
                                                                 fields_mapping={
                                                                     "BID": "bidBestLevel.price",
                                                                     "ASK": "askBestLevel.price"})
        self.subscribe_kafka_trades()

    def subscribe_kafka_trades(self) -> None:
        fields = {'portfolio': 'portfolio'}
        for isin in self.all_isins:
                self.global_subscription_service.subscribe_kafka(id=f"{isin}:SFMQ:Trade",
                                                                 symbol_filter=isin,
                                                                 topic=f"COALESCENT_DUMA.SFMQ.Trade",
                                                                 fields_mapping=fields)

    def update_LF(self) -> None:
        try:
            trades_to_publish = None
            for time_lag, trade_manager in self._trade_manager_dict.items():
                processed_trades = trade_manager.get_trades()
                if processed_trades.empty:
                    continue
                if trades_to_publish is None:
                    trades_to_publish = processed_trades
                    trades_to_publish.rename(columns={"lagged_spread_pl": f"self.lagged_spread_pl_{time_lag}"},
                                             inplace=True)
                else:
                    trades_to_publish[f"lagged_spread_pl_{time_lag}"] = processed_trades['lagged_spread_pl']
            if trades_to_publish is not None:
                self.publish_trades_on_dashboard(trades_to_publish)
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        if datetime.today().time() < dt.time(17, 29, 40):
            self.get_live_data()

    def on_trade(self, new_trades):
        # remove all trades with vtvfut as portfolio
        portfolios_trades = new_trades[new_trades["portfolio"] != "VTVFUT"]

        trades_to_publish = None
        for time_lag, trade_manager in self._trade_manager_dict.items():
            processed_new = trade_manager.on_trade(portfolios_trades)
            processed_trades = trade_manager.get_trades_to_publish(processed_new)
            if trades_to_publish is None:
                trades_to_publish = processed_trades
                trades_to_publish.rename(columns={"lagged_spread_pl": f"lagged_spread_pl_{time_lag}"}, inplace=True)
            else:
                trades_to_publish[f"lagged_spread_pl_{time_lag}"] = processed_trades['lagged_spread_pl']
        if not trades_to_publish.empty:
            self.publish_trades_on_dashboard(trades_to_publish)

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

    def get_live_data(self):
        raw = self.market_data.get_data_field(field=["BID", "ASK"])
        if raw is None or raw.empty:
            return
        snapshot = {isin: FairvaluePrice.scalar(isin, price) for isin, price in raw.mean(axis=1).items()}
        self.book_storage.append(snapshot)

    def on_stop(self):
        for trade_manager in self._trade_manager_dict.values():
            trade_manager.close()
