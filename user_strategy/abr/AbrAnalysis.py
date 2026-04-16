import datetime
import logging

from time import sleep
import datetime as dt

from market_monitor.publishers.rabbit_publisher import RabbitMessaging
from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.common.trade_manager.book_memory import FairvaluePrice, BookStorage
from market_monitor.strategy.common.trade_manager.trade_manager import TradeManager
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI


class AbrAnalysis(StrategyUI):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.rabbit_trade_dashboard_messaging: RabbitMessaging | None = None
        self.redis_trade_dashboard_messaging: RedisMessaging | None = None

        self.mid: dict[str, FairvaluePrice] = {}

        self.instruments = ["IT0005090300", "IT0003492391", "IT0025496222", "IT0003856405", "IT0004176001", "IT0005541336", "IT0005119810", "IT0000066123", "IT0003132476", "IT0000062957", "IT0005508921", "IT0003153415",
                            "IT0001347308","IT0003128367","IT0004776628","NL0015000LU4","IT0003796171","IT0005366767","IT0005599938","IT0001233417","IT0003497168","IT0000072170","IT0004056880","IT0000062072","IT0004965148",
                            "IT0003261697","IT0004810054","IT0005239360","IT0000072618","IT0005211237","NL0011585146","IT0005218380","IT0005495657","IT0004764699","IT0001250932","IT0003828271","NL0000226223","NL00150001Q9","LU2598331598","IT0003242622","NL0015435975"]

            # kwargs.get("instruments"))

        # -------------------------------------- BOOK & PRICE SECTION --------------------------------------------------

        self.book_storage: BookStorage = BookStorage(maxlen=100)

        rabbit_cfg = kwargs.get('rabbit_data_export', {})

        if rabbit_cfg.get('activate', False):
            rabbit_params = rabbit_cfg.get('rabbit_params', {})
            self.rabbit_trade_dashboard_messaging = RabbitMessaging(**rabbit_params)
            self.rabbit_exporting_channel = rabbit_cfg.get('channel', "rabbit_export_channel")
            sep = "=" * 80
            print(f"""{sep} \n ==== Rabbit export -> channel: {self.rabbit_exporting_channel} ==== \n {sep}""".strip())
        self.trade_manager = TradeManager(self.book_storage, **kwargs["trade_manager"])

    def wait_for_book_initialization(self):
        while dt.datetime.today().time() < dt.time(9, 5):
            return False
        return True
        # while True:
        #     data = self.market_data.get_data_field(field=["BID", "ASK"])
        #     if data is not None and not data.empty:
        #         break
        #     sleep(1)
        # self.on_start_strategy()
        # return True

    def on_start_strategy(self):
        pass

    def on_market_data_setting(self) -> None:
        self.subscribe_kafka()

    def subscribe_kafka(self):

        """"
        per sottoscrizioni Kafka chiedi a me
        """

        for isin in self.instruments:
            self.global_subscription_service.subscribe_kafka(
                id=f"{isin}_book",
                symbol_filter=isin,
                topic=f"COALESCENT_DUMA.MTAA.BookBest",
                fields_mapping={"ASK":"askBestLevel.price", "BID": "bidBestLevel.price" }
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
        if len(new_trades.index) > 1000:
            return
        new_trades.drop('side',axis=1, inplace=True, errors='ignore')
        self.trade_manager.on_trade(new_trades)
        trades_to_publish = self.trade_manager.get_trades_to_publish()
        self.publish_trades_on_dashboard(trades_to_publish)

    def publish_trades_on_dashboard(self, new_trades):

        if self.rabbit_trade_dashboard_messaging:
            self.rabbit_trade_dashboard_messaging.export_message(channel=self.rabbit_exporting_channel,
                                                                 value=new_trades,
                                                                 date_format='iso',
                                                                 orient="records")

        """
        Per usare Redis devi installare il programma (è in crossmarket)
        """

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
       self._instruments = value

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments
