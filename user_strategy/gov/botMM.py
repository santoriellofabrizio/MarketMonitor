import datetime
import logging
from collections import defaultdict
from pathlib import Path
from time import sleep

import pandas as pd
import datetime as dt

from market_monitor.publishers.rabbit_publisher import RabbitMessaging
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.common.trade_manager.book_memory import FairvaluePrice, BookStorage
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI


class botMM(StrategyUI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._instruments_meta = None
        self.rabbit_trade_dashboard_messaging: RabbitMessaging | None = None
        self.redis_trade_dashboard_messaging: RedisMessaging | None = None

        path_str = kwargs.get("instruments_list_path")
        excel_path = Path(path_str) if path_str else None

        self.id_for_each_isin = defaultdict(list)
        self.mid: dict[str, FairvaluePrice] = {}

        self.instruments = kwargs.get("instruments")

        # -------------------------------------- BOND SECTION ----------------------------------------------------------
        self.bond_markets = ["MTSC"]
        self.bond_isins = self.instruments

        # -------------------------------------- BOOK & PRICE SECTION --------------------------------------------------
        self.book_storage: BookStorage = BookStorage()

        rabbit_cfg = kwargs.get('rabbit_data_export', {})

        if rabbit_cfg.get('activate', False):
            rabbit_params = rabbit_cfg.get('rabbit_params', {})
            self.rabbit_trade_dashboard_messaging = RabbitMessaging(**rabbit_params)
            self.rabbit_exporting_channel = rabbit_params.get('channel_rabbit', "rabbit_export_channel")
            sep = "=" * 80
            print(f"""{sep} \n ==== Rabbit export -> channel: {self.rabbit_exporting_channel} ==== \n {sep}""".strip())
        self.trade_manager = TradeManager(self.book_storage, **kwargs["trade_manager"])

    def wait_for_book_initialization(self):
        while dt.datetime.today().time() < dt.time(8, 50):
            return False
        while True:
            data = self.market_data.get_data_field(field=["BID", "ASK"])
            if data is not None and not data.empty:
                break
            sleep(1)
        self.on_start_strategy()
        return True

    def on_start_strategy(self):

        new_trades = self.trade_manager.get_trades()
        new_trades["description"] = new_trades["isin"].map(lambda i: self._instruments_meta[i]["description"])
        new_trades["price_multiplier"] = new_trades["isin"].map(lambda i: self._instruments_meta[i]["multiplier"])
        new_trades["reference_ester"] = new_trades["isin"].map(lambda i: self._instruments_meta[i]["reference_ester"])
        if not new_trades.empty:
            self.publish_trades_on_dashboard(new_trades)

    def on_market_data_setting(self) -> None:
        self.subscribe_kafka()

    def subscribe_kafka(self):

        for isin in self.bond_isins:
            self.global_subscription_service.subscribe_kafka(
                id=f"{isin}_book",
                symbol_filter=isin,
                topic=f"COALESCENT_DUMA.SSOB.BookByLevels",
                fields_mapping={"ASK":"askLevels.0.price", "BID": "bidLevels.0.price" }
            )
            for ev in ("PublicDeal", "Trade"):
                self.global_subscription_service.subscribe_trades_kafka(
                    id=f"MTSC_{isin}:{ev}",
                    symbol_filter=isin,
                    topic=f"COALESCENT_DUMA.MTSC.{ev}"

                )

    def update_LF(self) -> None:
        try:
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        if datetime.datetime.today().time() < dt.time(17, 29, 40):
            self.get_live_data()

    def on_trade(self, new_trades):
        print(new_trades)

        if isinstance(new_trades, pd.DataFrame) and not new_trades.empty:
            new_trades["description"] = new_trades["isin"].map(lambda i: self._instruments_meta[i]["description"])
            new_trades["price_multiplier"] = new_trades["isin"].map(lambda i: self._instruments_meta[i]["multiplier"])
            new_trades["reference_ester"] = new_trades["isin"].map( lambda i: self._instruments_meta[i]["reference_ester"])

        self.trade_manager.on_trade(new_trades)
        trades_to_publish = self.trade_manager.get_trades_to_publish()
        self.publish_trades_on_dashboard(trades_to_publish)

    def publish_trades_on_dashboard(self, new_trades):

        if self.rabbit_trade_dashboard_messaging:
            self.rabbit_trade_dashboard_messaging.export_message(channel=self.rabbit_exporting_channel,
                                                                 value=new_trades,
                                                                 date_format='iso',
                                                                 orient="records")

    def get_live_data(self):
        book = self.market_data.get_data_field(field=["BID", "ASK"])
        if book is None or book.empty:
            return

        # scarta righe senza book ancora
        book = book.dropna(subset=["BID", "ASK"])
        if book.empty:
            return

        mid = book.mean(axis=1)

        for instrument_id, price in mid.items():
            isin, _ = instrument_id.split("_")
            self.mid[isin] = FairvaluePrice.scalar(isin, price)

        self.book_storage.append(dict(self.mid))

    def on_stop(self):
        self.trade_manager.close()


    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        # value è ora lista di dict dal YAML:
        # [{"isin": "IT...", "description": "...", "reference_ester": ..., "multiplier": ...}, ...]
        if value and isinstance(value[0], dict):
            self._instruments_meta: dict[str, dict] = {
                item["isin"]: {
                    "description": item.get("description"),
                    "reference_ester": item.get("reference_ester"),
                    "multiplier": item.get("multiplier"),
                }
                for item in value
            }
            self._instruments = list(self._instruments_meta.keys())
        else:
            # fallback: lista di stringhe plain (retrocompatibilità)
            self._instruments_meta = {}
            self._instruments = value


    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments
