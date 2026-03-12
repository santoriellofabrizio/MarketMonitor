import datetime
import logging
import queue
import threading
from enum import Enum
from typing import Optional

import pandas as pd
import itertools


class TradeTyp(Enum):
    OWN = 1
    MARKET = 2


class AbstractTrade:
    _id_generator = itertools.count(start=1)

    def __init__(self, ticker, isin, timestamp, quantity, price, market=None, currency=None, price_multiplier=1,
                 **extra):
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
        self.own_trade: bool | None = None
        self.is_elaborated: bool = False
        self.extra: dict = extra or {}

    def is_my_trade(self):
        return self.own_trade

    def time_since_trade(self) -> float:
        """Restituisce il tempo trascorso dal trade in secondi (float)."""
        # Usa UTC per evitare problemi di timezone
        now = datetime.datetime.now()

        # Se timestamp è naive, assumilo UTC
        ts = self.timestamp
        if isinstance(ts, int):
            ts = datetime.datetime.fromtimestamp(ts / 1_000_000_000)
        return (now - ts).total_seconds()


class MyTrade(AbstractTrade):

    def __init__(self, ticker, isin, timestamp, quantity, price, market, currency, side, price_multiplier, **extra):
        super().__init__(ticker, isin, timestamp, quantity, price, market, currency, price_multiplier, **extra)
        self.side = side
        self.own_trade = True


class Trade(AbstractTrade):
    def __init__(self, ticker, isin, timestamp, quantity, price, market, currency, price_multiplier, **extra):
        super().__init__(ticker, isin, timestamp, quantity, price, market, currency, price_multiplier, **extra)
        self.own_trade = False


class TradeFactory:

    def __init__(self):
        pass

    @staticmethod
    def build_trades_obj(trades: pd.DataFrame):
        known_cols = {'own_trade', 'isin', 'last_update', 'quantity', 'price', 'market', 'currency', 'price_multiplier'}
        extra_cols = [c for c in trades.columns if c not in known_cols]

        for row in trades.itertuples():
            ticker = row.Index
            extra = {col: getattr(row, col, None) for col in extra_cols}
            if row.own_trade != 0:
                side = 'bid' if row.own_trade == 1 else 'ask'
                yield MyTrade(ticker, row.isin, row.last_update, row.quantity, row.price,
                              row.market, row.currency, side,
                              getattr(row, "price_multiplier", 1), **extra)
            else:
                yield Trade(ticker, row.isin, row.last_update, row.quantity, row.price,
                            row.market, row.currency,
                            getattr(row, "price_multiplier", 1), **extra)


class TradeStorage:

    def __init__(self):
        self.lock = threading.Lock()
        self._storage = {}  # Contiene tutti i trades
        self.trade_to_elaborate = queue.Queue()  # Coda dei trades da elaborare
        self._my_trades_indexes = []

    def get_last_trades(self, n: int | None = None):
        with self.lock:
            if n is None:
                return self._storage
            else:
                if isinstance(self._storage, dict):
                    storage = [*self._storage.values()]
                    return storage[-n:] if len(storage) > n else storage
                return self._storage[-n:] if len(self._storage) > n else self._storage

    def add_trade(self, trade: AbstractTrade):
        with self.lock:
            if trade.time_since_trade() < 30:
                if trade.is_my_trade():
                    self._my_trades_indexes.append(trade.trade_index)
                self.trade_to_elaborate.put(trade.trade_index)
            self._storage[trade.trade_index] = trade

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

    def get_trades_by_index(self, index: int) -> AbstractTrade | None:
        trade = self._storage.get(index)
        if trade is None:
            logging.error(f"Index {index} not found in storage ({len(self._storage)} entries)")
        return trade

    def set_trade_as_elaborated(self, trade: AbstractTrade):
        with self.lock:
            # Aggiorna lo stato del trade
            trade.is_elaborated = True
            self._storage[trade.trade_index] = trade
