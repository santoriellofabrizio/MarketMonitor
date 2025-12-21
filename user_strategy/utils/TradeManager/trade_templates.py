import datetime
import logging
import queue
import threading
import time
from typing import Optional

import pandas as pd
import itertools


class AbstractTrade:
    _id_generator = itertools.count(start=0)

    def __init__(self, ticker, isin, timestamp, quantity, price, market=None, currency=None, price_multiplier=1):
        self.trade_index: int = next(self._id_generator)
        self.ticker: str = ticker
        self.isin: str = isin
        self.quantity: int = quantity
        self.timestamp = timestamp
        self.price: float = price
        self.price_multiplier: float = price_multiplier
        self.ctv = self.price * self.quantity * self.price_multiplier
        self.market: Optional[str] = market
        self.currency: Optional[str] = currency
        self.side = None
        self.spread_pl: float | None = None
        self.spread_pl_model: float | None = None
        self.lagged_spread_pl: float | None = None
        self.lagged_spread_pl_model: float | None = None
        self.own_trade: bool | None = None
        self.is_elaborated: bool = False

    def is_my_trade(self):
        return self.own_trade

    def time_since_trade(self) -> float:
        """Restituisce il tempo trascorso dal trade in secondi (float)."""
        # Usa UTC per evitare problemi di timezone
        now = datetime.datetime.now()

        # Se timestamp è naive, assumilo UTC
        ts = self.timestamp
        return (now - ts).total_seconds()


class MyTrade(AbstractTrade):

    def __init__(self, ticker, isin, timestamp, quantity, price, market, currency, side, price_multiplier):
        super().__init__(ticker, isin, timestamp, quantity, price, market, currency, price_multiplier)
        self.side = side
        self.own_trade = True


class Trade(AbstractTrade):
    def __init__(self, ticker, isin, timestamp, quantity, price, market, currency, price_multiplier):
        super().__init__(ticker, isin, timestamp, quantity, price, market, currency, price_multiplier)
        self.own_trade = False


class TradeFactory:

    def __init__(self):
        pass

    @staticmethod
    def build_trades_obj(trades: pd.DataFrame):
        for row in trades.itertuples():
            ticker = row.Index
            if row.own_trade != 0:
                side = 'bid' if row.own_trade == 1 else 'ask'
                yield MyTrade(ticker, row.isin, row.last_update, row.quantity, row.price, row.market, row.currency,
                              side, getattr(row, "price_multiplier", 1))
            else:
                yield Trade(ticker, row.isin, row.last_update, row.quantity, row.price, row.market, row.currency,
                            getattr(row, "price_multiplier", 1))


class TradeStorage:

    def __init__(self):
        self.lock = threading.Lock()
        self._storage = []  # Contiene tutti i trades
        self.trade_to_elaborate = queue.Queue()  # Coda dei trades da elaborare
        self._my_trades_indexes = []

    def get_last_trades(self, n: int | None = None):
        with self.lock:
            if n is None:
                return self._storage
            else:
                return self._storage[-n:] if len(self._storage) > n else self._storage

    def add_trade(self, trade: AbstractTrade):
        with self.lock:
            if trade.time_since_trade() < 30:
                if trade.is_my_trade():
                    self._my_trades_indexes.append(trade.trade_index)
                self.trade_to_elaborate.put(trade.trade_index)
            self._storage.append(trade)


    def append(self, value):
        self._storage.append(value)

    def get_trade_index_to_elaborate(self, timeout=None):
        """
        Attende fino al timeout per ottenere l'indice del trade da elaborare
        e restituisce il trade corrispondente.
        """
        try:
            index = self.trade_to_elaborate.get(timeout=timeout)
            return index
        except queue.Empty:
            return None

    def get_trades_by_index(self, index: list[int] | int):
        while True:
            try:
                return self._storage[index]
            except IndexError:
                logging.error(f"Index {index} not found in storage ({len(self._storage)} long)")
            time.sleep(0.1)

    def set_trade_as_elaborated(self, trade: AbstractTrade):
        with self.lock:
            # Aggiorna lo stato del trade
            trade.is_elaborated = True
            self._storage[trade.trade_index] = trade
