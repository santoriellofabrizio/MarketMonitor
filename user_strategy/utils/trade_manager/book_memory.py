"""
Storage semplice per snapshot del book con API chiara.
"""

from collections import deque
from datetime import datetime
from typing import Optional
import pandas as pd


class BookStorage:
    """
    Storage per snapshot temporali del book (timestamp + mid prices).

    Usage:
        >>> book = BookStorage(maxlen=3)
        >>> book.append(mid_prices_series)
        >>> mid = book.get_mid("IE00B4L5Y983")
        >>> old_mid = book.get_mid("IE00B4L5Y983", old=True)
    """

    def __init__(self, maxlen: int = 3):
        self._storage: deque[tuple[datetime, pd.Series]] = deque(maxlen=maxlen)

    def append(self, mid_prices: pd.Series, time_snapshot: Optional[None] = None) -> None:
        """Aggiungi nuovo snapshot."""
        time_snapshot = time_snapshot or datetime.now()
        self._storage.append((time_snapshot, mid_prices.copy()))

    def get_mid(self, isin: str, old: bool = False) -> Optional[float]:
        """
        Get mid price per ISIN.

        Args:
            isin: ISIN del security
            old: Se True usa oldest snapshot (index 0), altrimenti newest (-1)

        Returns:
            Mid price o None se non trovato
        """
        if not self._storage:
            return None

        index = 0 if old else -1
        timestamp, mid_prices = self._storage[index]
        return mid_prices.get(isin)

    def get_age_seconds(self, old: bool = False) -> Optional[float]:
        """Get età dello snapshot in secondi."""
        if not self._storage:
            return None

        index = 0 if old else -1
        timestamp, _ = self._storage[index]
        return (datetime.now() - timestamp).total_seconds()

    def __len__(self) -> int:
        return len(self._storage)

    def __bool__(self) -> bool:
        return len(self._storage) > 0

    def get_last_before(self, timestamp: datetime) -> Optional[tuple[datetime, pd.Series]]:
        """
        Get l'ultimo snapshot prima del timestamp dato.

        Args:
            timestamp: Timestamp di riferimento

        Returns:
            Tuple (timestamp, mid_prices) o None se non trovato
        """
        if isinstance(timestamp, int):
            timestamp = datetime.fromtimestamp(timestamp/ 1_000_000_000)

        for ts, mid_prices in reversed(self._storage):
            if ts <= timestamp:
                return (ts, mid_prices)
        return None

    def get_first_after(self, timestamp: datetime) -> Optional[tuple[datetime, pd.Series]]:
        """
        Get il primo snapshot dopo il timestamp dato.

        Args:
            timestamp: Timestamp di riferimento

        Returns:
            Tuple (timestamp, mid_prices) o None se non trovato
        """
        for ts, mid_prices in self._storage:
            if ts >= timestamp:
                return (ts, mid_prices)
        return None

    def __getitem__(self, index: int) -> tuple[datetime, pd.Series]:
        """Backward compatibility: accesso diretto by index."""
        return self._storage[index]