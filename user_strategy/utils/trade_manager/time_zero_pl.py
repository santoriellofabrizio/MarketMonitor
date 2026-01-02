import datetime
import logging
import threading
import time
import pandas as pd
from user_strategy.utils.trade_manager.trade_templates import AbstractTrade, TradeStorage

logger = logging.getLogger(__name__)


class TimeZeroPLManager(threading.Thread):

    def __init__(self, trade_storage: TradeStorage,
                 mid_price_storage: pd.Series,
                 model_price: pd.Series | None = None,
                 time_zero_lag: float = 10.):
        # IMPORTANTE: Cambiato nome per evitare conflitti interni di threading
        super().__init__(name="TimeZeroPLThread", daemon=True)
        self.trade_storage = trade_storage
        self.mid_price_storage = mid_price_storage
        self.model_price = model_price
        self.time_zero_lag = time_zero_lag

        # Flag per la chiusura
        self._is_running = True
        self._stop_event = threading.Event()

        logger.info(f"Inizializzato con time_zero_lag = {self.time_zero_lag}s")

    def stop(self):
        """Metodo chiamato esternamente per chiudere il thread."""
        logger.info("Ricevuto segnale di stop per TimeZeroPLThread")
        self._is_running = False
        self._stop_event.set()  # Sveglia il thread se sta dormendo

    def run(self) -> None:
        logger.info("Thread avviato.")
        while self._is_running:
            # Se get_trade_index_to_elaborate è bloccante, assicurati che abbia un timeout
            trade_index = self.trade_storage.get_trade_index_to_elaborate()

            # Se il manager dello storage restituisce None o un segnale di stop
            if trade_index is None:
                if not self._is_running: break
                self._stop_event.wait(1.0)  # Aspetta un secondo prima di riprovare
                continue

            trade = self.trade_storage.get_trades_by_index(trade_index)
            if trade:
                # Se process_trade restituisce False, significa che dobbiamo chiudere
                if not self.process_trade(trade):
                    break
            else:
                self._stop_event.wait(0.5)

    def process_trade(self, trade: AbstractTrade) -> bool:
        """Restituisce True se completato, False se interrotto dallo stop."""
        trade_timestamp = trade.timestamp
        logger.info(f"Elaborazione trade {trade.trade_index}...")

        while self._is_running:
            diff_seconds = (datetime.datetime.now() - trade_timestamp).total_seconds()

            if diff_seconds > self.time_zero_lag * 4:
                self.trade_storage.set_trade_as_elaborated(trade)
                return True
            elif diff_seconds > self.time_zero_lag:
                self._calculate_time_zero_pl(trade)
                self.trade_storage.set_trade_as_elaborated(trade)
                return True
            else:
                wait_time = self.time_zero_lag - diff_seconds
                logger.info(f"Attendo {wait_time:.1f}s per trade {trade.trade_index}...")

                # Invece di time.sleep(max(wait_time, 1))
                # Aspetta wait_time, ma se stop() viene chiamato, si sveglia subito
                interrupted = self._stop_event.wait(timeout=max(wait_time, 0.1))
                if interrupted or not self._is_running:
                    return False
        return False

    def _calculate_time_zero_pl(self, trade: AbstractTrade):

        mid_price, time_snip = self.get_mid(trade.isin)
        logger.info(
            f"[Trade {trade.trade_index} {trade.isin} time: {trade.timestamp}]"
            f" calculating pl with mid {mid_price} snipped at {time_snip}")
        trade.lagged_spread_pl = self.calculate_time_zero_pl(trade, mid_price)

        if self.model_price is not None:
            model_price = self.get_model_price(trade.isin)
            trade.lagged_spread_pl_model = self.calculate_time_zero_pl(trade, model_price)

    def get_mid(self, isin: str):
        """Recupera il prezzo di riferimento (mid price) per l'ISIN specificato."""
        try:
            if not len(self.mid_price_storage):
                return None, None
            time_snip, storage = self.mid_price_storage[-1]
            mid_price = storage[isin]
            logger.info(f"Mid price per {isin}: {mid_price}. snipped at: {time_snip}")
            return mid_price, time_snip
        except Exception as e:
            logger.debug(f"ISIN {isin} non trovato book, o storage vuoto", exc_info=e)
            return None, None

    def get_model_price(self, isin: str):
        try:
            mid_price = self.model_price.get(isin, None)
            logger.info(f"Model price per {isin}: {mid_price}.")
            return mid_price
        except KeyError:
            logger.debug(f"ISIN {isin} non trovato book. N")
            return None

    @staticmethod
    def calculate_time_zero_pl(trade: AbstractTrade, price: float):

        if (price is None) or price <= 0:
            return None

        logger.info(f"Calcolo mid PL per trade {trade.trade_index}.")
        trade_price = trade.price
        qty = trade.quantity

        side_map = {"bid": 1, "ask": -1}
        side = side_map.get(trade.side, 0)

        if price is None:
            logger.info(f"Mid price non trovato per isin {trade.isin}.")
            return
        pl = (price - trade_price) * qty * side
        logger.info(f"PL calcolato per il trade {trade.trade_index}: {pl}.")
        return pl
