import datetime as dt
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Literal


import pandas as pd

from market_monitor.publishers.rabbit_publisher import RabbitMessaging
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.common.trade_manager.book_memory import BookStorage, BookSnapshot, FairvaluePrice
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from market_monitor.utils.book_utils import SpreadEWMA, PriceEWMA

logger = logging.getLogger(__name__)

class CreditTradeAnalysis(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.price_source: Literal['bloomberg', 'kafka'] = kwargs.get('price_source', 'kafka')
        self.book_filter = SpreadEWMA(**kwargs.pop("book_filter_params", {}))
        self.rabbit_trade_dashboard_messaging: RabbitMessaging | None = None
        self.redis_trade_dashboard_messaging: RedisMessaging | None = None

        path_str = kwargs.get("instruments_list_path")
        excel_path = Path(path_str) if path_str else None

        self.id_for_each_isin = defaultdict(list)
        self.mid: dict[str, FairvaluePrice] = {}

        if excel_path and excel_path.exists():
            self.instruments_df = pd.read_excel(excel_path).set_index("isin")
            self.all_isin = [i.upper().strip() for i in self.instruments_df.keys()]
        else:
            raise Exception("missing valid input")

        self.instruments = [f"{instr}_{ccy}" for instr in self.instruments_df.index for ccy in ['EUR', 'USD']]
        # -------------------------------------- ETF SECTION -----------------------------------------------------------
        self.etf_markets = ["ETFP", "XPAR", "XAMS"]
        self.etf_isins = list(self.instruments_df[self.instruments_df["type"] == "ETF"].index)
        # -------------------------------------- BOND SECTION ----------------------------------------------------------
        self.bond_markets = ["ETLX", "XMOT", "MOTX"]
        self.bond_isins = list(self.instruments_df[self.instruments_df["type"] == "BOND"].index)
        # -------------------------------------- FUTURE SECTION --------------------------------------------------------
        self.future_markets = ["XEUR"]
        self.future_isins = list(self.instruments_df[self.instruments_df["type"] == "FUTURE"].index)
        # -------------------------------------- TRADE SECTION ---------------------------------------------------------
        self.trade_isin_multiplier_mapping = self.instruments_df["multiplier"].to_dict()
        self.trade_isin_description_mapping = self.instruments_df["description"].to_dict()
        # -------------------------------------- BOOK & PRICE SECTION --------------------------------------------------
        self.book_storage: BookStorage = BookStorage()

        self.book_filter = SpreadEWMA(**kwargs.pop("book_filter_params", {}))
        # -------------------------------------- SETTING INSTRUMENTS ---------------------------------------------------
        self.columns_dashboard = kwargs.get("columns_dashboard")

        rabbit_cfg = kwargs.get('rabbit_data_export', {})

        if rabbit_cfg.get('activate', False):
            rabbit_params = rabbit_cfg.get('rabbit_params', {})
            self.rabbit_trade_dashboard_messaging = RabbitMessaging(**rabbit_params)
            self.rabbit_exporting_channel = rabbit_cfg.get('channel', "rabbit_export_channel")
            sep = "=" * 80
            print(f"""{sep} \n ==== Rabbit export -> channel: {self.rabbit_exporting_channel} ==== \n {sep}""".strip())

        redis_export_cfg = kwargs.get('redis_data_export', {})

        if redis_export_cfg.get('activate', False):
            redis_params = redis_export_cfg.get('redis_export_params', {})
            self.redis_trade_dashboard_messaging = RedisMessaging(**redis_params)
            self.redis_exporting_channel = redis_export_cfg.get('channel', "redis_export_channel")
            sep = "=" * 80
            print(f"""{sep} \n ==== Redis export -> channel: {self.redis_exporting_channel} ==== \n {sep}""".strip())

        self.trade_manager = TradeManager(self.book_storage, **kwargs["trade_manager"])

    def wait_for_book_initialization(self):
        while datetime.today().time() < dt.time(8, 50):
            return False
        while True:
            data = self.market_data.get_data_field(field=["BID", "ASK"])
            if data is not None and not data.empty:
                break
            sleep(1)
        self.on_start_strategy()
        return True

    def on_start_strategy(self):

        for market in self.etf_markets:
            for etf in self.etf_isins:
                self.id_for_each_isin[etf].append(f"{market}_{etf}")

        for market in self.bond_markets:
            for bond in self.bond_isins:
                currency = self.market_data.currency_information.get(bond, "EUR")
                self.id_for_each_isin[f"{bond}_EUR"].append(f"{market}_{bond}")

        last_trades = self.trade_manager.get_trades()
        if not last_trades.empty:
            self.publish_trades_on_dashboard(last_trades)

    def on_market_data_setting(self) -> None:
        # Subscribe to original channel names with market: prefix

        if self.price_source == 'kafka':
            self.subscribe_kafka()
        else:
            self.subscribe_bloomberg()

    def subscribe_bloomberg(self):

        markets_mapping = {
            "ETFP": "IM",
            "XAMS": "NA",
            "XPAR": "FP",
            "MOTX": "MILA",
            "ETLX": "ETLX"}

        for markets, symbols in (
                (self.etf_markets, self.etf_isins),
                (self.bond_markets, self.bond_isins),
        ):
            for m in markets:
                for s in symbols:
                    if s in self.etf_isins:
                        sub_string = f"{s} {markets_mapping[m]} EQUITY"
                    else:
                        sub_string = f"{s} ISIN@{markets_mapping[m]} Corp"

                    self.global_subscription_service.subscribe_bloomberg(
                        id=f"{m}_{s}",
                        subscription_string=sub_string,
                    )

    def subscribe_kafka(self):
        fields = {
            "BID": "bidBestLevel.price",
            "ASK": "askBestLevel.price",
            "BID_SIZE": "bidBestLevel.quantity",
            "ASK_SIZE": "askBestLevel.quantity",
        }

        for markets, symbols in (
                (self.etf_markets, self.etf_isins),
                (self.bond_markets, self.bond_isins),
                (self.future_markets, self.future_isins),
        ):
            for m in markets:
                base = f"COALESCENT_DUMA.{m}"
                for s in symbols:
                    self.global_subscription_service.subscribe_kafka(
                        id=f"{m}_{s}",
                        symbol_filter=s,
                        topic=f"{base}.BookBest",
                        fields_mapping=fields,
                    )
                    for ev in ("PublicDeal", "Trade"):
                        self.global_subscription_service.subscribe_trades_kafka(
                            id=f"{m}_{s}:{ev}",
                            symbol_filter=s,
                            topic=f"{base}.{ev}"

                        )

    def update_LF(self) -> None:
        try:
            self.publish_trades_on_dashboard(self.trade_manager.get_trades())
        except Exception as e:
            logging.error(e)

    def update_HF(self):
        if datetime.today().time() < dt.time(17, 29, 40):
            self.get_live_data()

    def on_trade(self, new_trades):
        new_trades["price_multiplier"] = new_trades["isin"].map(self.trade_isin_multiplier_mapping)
        new_trades["description"] = new_trades["isin"].map(self.trade_isin_description_mapping)
        self.trade_manager.on_trade(new_trades)
        trades_to_publish = self.trade_manager.get_trades_to_publish()
        self.publish_trades_on_dashboard(trades_to_publish)

    def publish_trades_on_dashboard(self, new_trades):

        if self.redis_trade_dashboard_messaging:
            self.redis_trade_dashboard_messaging.export_message(channel=self.redis_exporting_channel,
                                                                value=new_trades,
                                                                date_format='iso',
                                                                orient="records")
        if self.rabbit_trade_dashboard_messaging:
            self.rabbit_trade_dashboard_messaging.export_message(channel=self.rabbit_exporting_channel,
                                                                 value=new_trades,
                                                                 date_format='iso',
                                                                 orient="records")

    def get_live_data(self):
        raw = self.market_data.get_data_field(field=["BID", "ASK"])
        if raw is None or raw.empty:
            return

        # scarta righe senza book ancora
        raw = raw.dropna(subset=["BID", "ASK"])
        if raw.empty:
            return

        self.book_filter.update(raw)
        valid = self.book_filter.get_valid_book(raw)
        if valid.empty:
            return

        mid = valid.mean(axis=1)

        grouped = {}

        for instrument_id, price in mid.items():
            _, isin = instrument_id.split("_", 1)
            ccy = self.market_data.currency_information.get(instrument_id, "EUR")

            if isin not in grouped:
                grouped[isin] = {}

            grouped[isin][ccy] = price
        for isin, ccy_prices in grouped.items():
            if isin not in self.mid:
                self.mid[isin] = FairvaluePrice.by_currency(isin, {})
            for currency, prices in ccy_prices.items():
                self.mid[isin]._prices[currency] = sum(prices) / len(prices) if isinstance(prices, dict) else prices

        self.book_storage.append(dict(self.mid))

    def on_stop(self):
        self.trade_manager.close()

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = value

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments

