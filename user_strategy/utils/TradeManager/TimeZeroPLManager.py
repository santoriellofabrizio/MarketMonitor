import datetime
import logging
import threading
import time
import pandas as pd
from user_strategy.utils.TradeManager.TradeClassTemplate import AbstractTrade, TradeStorage


class TimeZeroPLManager(threading.Thread):

    def __init__(self, trade_storage: TradeStorage,
                 mid_price_storage: pd.Series,
                 model_price: pd.Series | None = None,
                 time_zero_lag: float = 10.):
        super().__init__(name="TimeZeroPLThread", daemon=True)
        self.trade_storage = trade_storage
        self.mid_price_storage = mid_price_storage
        self.model_price = model_price
        self.time_zero_lag = time_zero_lag
        logging.info(f"Inizializzato con time_zero_lag = {self.time_zero_lag}s")

    def run(self) -> None:
        logging.info("Thread avviato.")
        while True:
            logging.info("Attesa trade da elaborare...")
            trade_index = self.trade_storage.get_trade_index_to_elaborate()
            logging.info(f"Trade da elaborare: indice {trade_index}")
            trade = self.trade_storage.get_trades_by_index(trade_index)
            if trade:
                self.process_trade(trade)
            else:
                logging.info("Trade non trovato. Pulisco evento e aspetto nuovi trade.")

    def process_trade(self, trade: AbstractTrade):
        trade_timestamp = trade.timestamp
        logging.info(f"Elaborazione trade {trade.trade_index}...")

        while True:
            diff_seconds = (datetime.datetime.now() - trade_timestamp).seconds
            if diff_seconds > self.time_zero_lag * 4:
                self.trade_storage.set_trade_as_elaborated(trade)
            elif diff_seconds > self.time_zero_lag:

                self._calculate_time_zero_pl(trade)
                self.trade_storage.set_trade_as_elaborated(trade)
                break
            else:
                wait_time = self.time_zero_lag - diff_seconds
                logging.info(f"Attendo {wait_time}s per trade {trade.trade_index}...")
                time.sleep(max(wait_time, 1))

    def _calculate_time_zero_pl(self, trade: AbstractTrade):

        mid_price, time_snip = self.get_mid(trade.isin)
        logging.info(
            f"[Trade {trade.trade_index} {trade.isin} time: {trade.timestamp}]"
            f" calculating pl with mid {mid_price} snipped at {time_snip}")
        trade.lagged_spread_pl = self.calculate_time_zero_pl(trade, mid_price)

        if self.model_price is not None:
            model_price = self.get_model_price(trade.isin)
            trade.lagged_spread_pl_model = self.calculate_time_zero_pl(trade, model_price)

    def get_mid(self, isin: str):
        """Recupera il prezzo di riferimento (mid price) per l'ISIN specificato."""
        try:
            if not len(self.mid_price_storage): return None, None
            time_snip, storage = self.mid_price_storage[-1]
            mid_price = storage[isin]
            logging.info(f"Mid price per {isin}: {mid_price}. snipped at: {time_snip}")
            return mid_price, time_snip
        except Exception as e:
            logging.debug(f"ISIN {isin} non trovato book, o storage vuoto", exc_info=e)
            return None, None

    def get_model_price(self, isin: str):
        try:
            mid_price = self.model_price.get(isin, None)
            logging.info(f"Model price per {isin}: {mid_price}.")
            return mid_price
        except KeyError:
            logging.debug(f"ISIN {isin} non trovato book. N")
            return None

    @staticmethod
    def calculate_time_zero_pl(trade: AbstractTrade, price: float):

        if (price is None) or price <= 0:
            return None

        logging.info(f"Calcolo mid PL per trade {trade.trade_index}.")
        trade_price = trade.price
        qty = trade.quantity

        side_map = {"bid": 1, "ask": -1}
        side = side_map.get(trade.side, 0)

        if price is None:
            logging.info(f"Mid price non trovato per isin {trade.isin}.")
            return
        pl = (price - trade_price) * qty * side
        logging.info(f"PL calcolato per il trade {trade.trade_index}: {pl}.")
        return pl


