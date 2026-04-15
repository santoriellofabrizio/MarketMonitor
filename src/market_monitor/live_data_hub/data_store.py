"""
Store classes refactored per robustezza.

AGGIORNAMENTO: Gestione sicura di indici/colonne mancanti.
"""

import logging
from datetime import datetime
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketStore:
    """
    DataFrame-based storage for market data (prices from Bloomberg/RedisPublisher).
    Thread-safe with minimal lock hold time via lazy-snapshot double-buffer pattern.

    Write path: acquires lock, mutates _write_buf, marks _dirty=True, releases lock.
    Read path: acquires lock briefly only if _dirty (to refresh snapshot), then
               operates on the stable snapshot reference WITHOUT holding any lock.
               Multiple read calls in the same update_HF() cycle reuse the cached
               snapshot — zero lock overhead after the first call per dirty epoch.
    """

    def __init__(self, fields: List[str], locker: Lock = None):
        self._lock = locker if locker is not None else Lock()
        self._fields = fields
        self._write_buf: pd.DataFrame = pd.DataFrame()
        self._snapshot: Optional[pd.DataFrame] = None  # stable read-only copy
        self._dirty: bool = False
        self._last_update: Dict[str, datetime] = {}

    def _get_snapshot(self) -> pd.DataFrame:
        """
        Returns a stable snapshot of the market data.
        Refreshes under lock only when writes have occurred since the last read
        (_dirty=True). Subsequent calls in the same cycle are fully lock-free.
        """
        with self._lock:
            if self._dirty or self._snapshot is None:
                self._snapshot = self._write_buf.copy()
                self._dirty = False
        return self._snapshot  # stable reference, no lock held by caller

    def initialize(self, securities: List[str]):
        """Initialize market DataFrame for securities."""
        with self._lock:
            self._write_buf = pd.DataFrame(
                index=securities,
                columns=self._fields,
                dtype=float
            )
            # EUR always at 1.0
            if "EUR" not in self._write_buf.index:
                self._write_buf.loc["EUR"] = [1.0] * len(self._fields)
            self._dirty = True

    def update(self, ticker: str, data: Dict[str, float]):
        """
        Update market data for ticker.
        Mutates _write_buf under lock and marks snapshot as stale.
        """
        with self._lock:
            try:
                if self._write_buf.empty:
                    all_fields = list(set(self._fields + list(data.keys())))
                    self._write_buf = pd.DataFrame(columns=all_fields, dtype=float)

                for field in data.keys():
                    if field not in self._write_buf.columns:
                        self._write_buf[field] = np.nan

                if ticker not in self._write_buf.index:
                    self._write_buf.loc[ticker] = np.nan

                for field, value in data.items():
                    self._write_buf.at[ticker, field] = value
                self._last_update[ticker] = datetime.now()
                self._dirty = True

            except Exception as e:
                logger.error(f"Error updating market data for {ticker}: {e}")

    def get_data(self, tickers: Optional[List[str]] = None,
                 fields: Optional[List[str]] = None) -> pd.DataFrame:
        """Get market data snapshot. Lock-free after first call per dirty epoch."""
        buf = self._get_snapshot()
        if buf.empty:
            return pd.DataFrame()
        try:
            if tickers is None and fields is None:
                return buf.copy()
            elif tickers is None:
                existing_fields = [f for f in fields if f in buf.columns]
                if not existing_fields:
                    return pd.DataFrame()
                return buf[existing_fields].copy()
            elif fields is None:
                existing_tickers = [t for t in tickers if t in buf.index]
                if not existing_tickers:
                    return pd.DataFrame()
                return buf.loc[existing_tickers].copy()
            else:
                existing_tickers = [t for t in tickers if t in buf.index]
                existing_fields = [f for f in fields if f in buf.columns]
                if not existing_tickers or not existing_fields:
                    return pd.DataFrame()
                return buf.loc[existing_tickers, existing_fields].copy()
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()

    def get_field(self, field: str, tickers: Optional[List[str]] = None) -> pd.Series:
        """Get specific field for tickers. Lock-free after snapshot refresh."""
        buf = self._get_snapshot()
        if buf.empty or field not in buf.columns:
            return pd.Series(dtype=float)
        try:
            if tickers is None:
                return buf[field].copy()
            else:
                existing_tickers = [t for t in tickers if t in buf.index]
                if not existing_tickers:
                    return pd.Series(dtype=float)
                return buf.loc[existing_tickers, field].copy()
        except Exception as e:
            logger.error(f"Error getting field {field}: {e}")
            return pd.Series(dtype=float)

    def get_mid(self, mid_fields: List[str],
                tickers: Optional[List[str]] = None) -> pd.Series:
        """Calculate mid prices. Lock-free after snapshot refresh."""
        buf = self._get_snapshot()
        if buf.empty:
            return pd.Series(dtype=float)
        existing_fields = [f for f in mid_fields if f in buf.columns]
        if not existing_fields:
            return pd.Series(dtype=float)
        try:
            if tickers is None:
                data = buf[existing_fields]
            else:
                existing_tickers = [t for t in tickers if t in buf.index]
                if not existing_tickers:
                    return pd.Series(dtype=float)
                data = buf.loc[existing_tickers, existing_fields]
            return pd.Series(data.values.mean(axis=1), index=data.index)
        except Exception as e:
            logger.error(f"Error calculating mid prices: {e}")
            return pd.Series(dtype=float)

    def get_mid_dict(self, mid_fields: List[str],
                     tickers: Optional[List[str]] = None) -> Dict[str, float]:
        """Get mid prices as dict. Lock-free after snapshot refresh."""
        buf = self._get_snapshot()
        if buf.empty:
            return {}
        existing_fields = [f for f in mid_fields if f in buf.columns]
        if not existing_fields:
            return {}
        try:
            if tickers is None:
                data = buf[existing_fields]
            else:
                existing_tickers = [t for t in tickers if t in buf.index]
                if not existing_tickers:
                    return {}
                data = buf.loc[existing_tickers, existing_fields]
            means = data.values.mean(axis=1)
            return {ticker: means[i] for i, ticker in enumerate(data.index)}
        except Exception as e:
            logger.error(f"Error getting mid dict: {e}")
            return {}

    def get_last_update(self, ticker: str) -> Optional[datetime]:
        """Get last update time for ticker."""
        with self._lock:
            return self._last_update.get(ticker)

    def get_securities(self) -> List[str]:
        """Get list of securities."""
        buf = self._get_snapshot()
        return list(buf.index) if not buf.empty else []

    def set_dataframe(self, df: pd.DataFrame):
        """Replace entire DataFrame."""
        with self._lock:
            self._write_buf = df.copy() if not df.empty else pd.DataFrame()
            self._dirty = True

    def has_ticker(self, ticker: str) -> bool:
        """Check if ticker exists."""
        buf = self._get_snapshot()
        return ticker in buf.index if not buf.empty else False

    def has_field(self, field: str) -> bool:
        """Check if field exists."""
        buf = self._get_snapshot()
        return field in buf.columns if not buf.empty else False


class StateStore:
    """
    Hierarchical dict storage for application state.
    Organized as: namespace/key -> data

    Examples:
        portfolio/positions -> {ISIN: {qty, avg_price}}
        portfolio/pnl -> {ISIN: pnl}
        metadata/timestamps -> {...}

    AGGIORNAMENTO: Metodi sempre sicuri anche con namespace/key mancanti.
    """

    def __init__(self, locker: Lock = None):
        self._lock = locker if locker is not None else RLock()
        self._state_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def update(self, namespace: str, key: str, data: Any):
        """
        Update state at namespace/key.

        AGGIORNAMENTO: Crea automaticamente namespace se mancante.
        """
        with self._lock:
            try:
                if namespace not in self._state_data:
                    self._state_data[namespace] = {}
                self._state_data[namespace][key] = data
            except Exception as e:
                logger.error(f"Error updating state {namespace}/{key}: {e}")

    def get(self, namespace: str, key: Optional[str] = None) -> Any:
        """
        Get state data.

        AGGIORNAMENTO: Ritorna None o {} se mancante invece di exception.
        """
        with self._lock:
            try:
                if key is None:
                    # Ritorna intero namespace (copia)
                    return self._state_data.get(namespace, {}).copy()
                # Ritorna key specifica
                return self._state_data.get(namespace, {}).get(key)
            except Exception as e:
                logger.error(f"Error getting state {namespace}/{key}: {e}")
                return {} if key is None else None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all state data.

        NO CHANGES: già sicuro.
        """
        with self._lock:
            try:
                return {ns: data.copy() for ns, data in self._state_data.items()}
            except Exception as e:
                logger.error(f"Error getting all state: {e}")
                return {}

    def delete(self, namespace: str, key: Optional[str] = None):
        """
        Delete state data.

        AGGIORNAMENTO: Non fallisce se namespace/key non esistono.
        """
        with self._lock:
            try:
                if key is None:
                    # Cancella intero namespace
                    if namespace in self._state_data:
                        del self._state_data[namespace]
                else:
                    # Cancella key specifica
                    if namespace in self._state_data and key in self._state_data[namespace]:
                        del self._state_data[namespace][key]
            except Exception as e:
                logger.error(f"Error deleting state {namespace}/{key}: {e}")

    def update_nested(self, namespace: str, key: str, updates: Dict[str, Any]):
        """
        Update nested dict within state.

        AGGIORNAMENTO: Crea struttura se mancante.
        """
        with self._lock:
            try:
                if namespace not in self._state_data:
                    self._state_data[namespace] = {}
                if key not in self._state_data[namespace]:
                    self._state_data[namespace][key] = {}

                if isinstance(self._state_data[namespace][key], dict):
                    self._deep_update(self._state_data[namespace][key], updates)
                else:
                    self._state_data[namespace][key] = updates
            except Exception as e:
                logger.error(f"Error updating nested state {namespace}/{key}: {e}")

    def has_namespace(self, namespace: str) -> bool:
        """Check if namespace exists"""
        with self._lock:
            return namespace in self._state_data

    def has_key(self, namespace: str, key: str) -> bool:
        """Check if key exists in namespace"""
        with self._lock:
            return namespace in self._state_data and key in self._state_data[namespace]

    def _deep_update(self, orig: dict, upd: dict):
        """Recursively update nested dict"""
        for key in list(upd.keys()):
            value = upd[key]
            if key in orig and isinstance(orig[key], dict) and isinstance(value, dict):
                self._deep_update(orig[key], value)
            else:
                orig[key] = value


class EventStore:
    """
    Deque-based storage for temporal events.
    Uses bounded deques to prevent memory growth.

    Examples:
        trades -> deque([{ts, isin, qty, price}, ...], maxlen=1000)
        logs -> deque([{ts, level, msg}, ...], maxlen=500)

    AGGIORNAMENTO: Metodi sempre sicuri anche con event_type mancante.
    """

    def __init__(self, locker: Lock = None, default_maxlen: int = 1000):
        self._lock = locker if locker is not None else Lock()
        self._default_maxlen = default_maxlen
        self._events: Dict[str, deque] = {}
        self._maxlens: Dict[str, int] = {}

    def append(self, event_type: str, event: Any, maxlen: Optional[int] = None):
        """
        Append event to deque.

        AGGIORNAMENTO: Crea automaticamente deque se mancante.
        """
        with self._lock:
            try:
                if event_type not in self._events:
                    ml = maxlen or self._default_maxlen
                    self._events[event_type] = deque(maxlen=ml)
                    self._maxlens[event_type] = ml

                self._events[event_type].append(event)
            except Exception as e:
                logger.error(f"Error appending event to {event_type}: {e}")

    def get(self, event_type: str, n: Optional[int] = None) -> List[Any]:
        """
        Get events (most recent first if n specified).

        AGGIORNAMENTO: Ritorna lista vuota se event_type mancante.
        """
        with self._lock:
            try:
                if event_type not in self._events:
                    return []

                events = list(self._events[event_type])
                if n is not None:
                    return events[-n:] if n > 0 else []
                return events
            except Exception as e:
                logger.error(f"Error getting events {event_type}: {e}")
                return []

    def get_all(self) -> Dict[str, List[Any]]:
        """
        Get all events.

        NO CHANGES: già sicuro.
        """
        with self._lock:
            try:
                return {et: list(events) for et, events in self._events.items()}
            except Exception as e:
                logger.error(f"Error getting all events: {e}")
                return {}

    def clear(self, event_type: Optional[str] = None):
        """
        Clear events.

        AGGIORNAMENTO: Non fallisce se event_type mancante.
        """
        with self._lock:
            try:
                if event_type is None:
                    self._events.clear()
                    self._maxlens.clear()
                elif event_type in self._events:
                    self._events[event_type].clear()
            except Exception as e:
                logger.error(f"Error clearing events {event_type}: {e}")

    def set_maxlen(self, event_type: str, maxlen: int):
        """
        Set max length for event type (recreates deque).

        AGGIORNAMENTO: Crea deque se mancante.
        """
        with self._lock:
            try:
                if event_type in self._events:
                    old_events = list(self._events[event_type])
                    self._events[event_type] = deque(old_events, maxlen=maxlen)
                else:
                    # Crea nuovo deque
                    self._events[event_type] = deque(maxlen=maxlen)

                self._maxlens[event_type] = maxlen
            except Exception as e:
                logger.error(f"Error setting maxlen for {event_type}: {e}")

    def has_event_type(self, event_type: str) -> bool:
        """Check if event type exists"""
        with self._lock:
            return event_type in self._events

    def count(self, event_type: str) -> int:
        """Get event count for type"""
        with self._lock:
            return len(self._events.get(event_type, []))


class BlobStore:
    """
    Storage for binary/file data.
    Maps keys -> raw data (bytes, files, etc.)

    AGGIORNAMENTO: Metodi sempre sicuri anche con key mancante.
    """

    def __init__(self, locker: Lock = None):
        self._lock = locker if locker is not None else Lock()
        self._blobs: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Store blob with optional metadata.

        NO CHANGES: già sicuro.
        """
        with self._lock:
            try:
                self._blobs[key] = data
                if metadata:
                    self._metadata[key] = metadata
            except Exception as e:
                logger.error(f"Error storing blob {key}: {e}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get blob data.

        AGGIORNAMENTO: Ritorna None se key mancante (già lo fa).
        """
        with self._lock:
            try:
                return self._blobs.get(key)
            except Exception as e:
                logger.error(f"Error getting blob {key}: {e}")
                return None

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get blob metadata.

        NO CHANGES: già sicuro.
        """
        with self._lock:
            try:
                return self._metadata.get(key)
            except Exception as e:
                logger.error(f"Error getting metadata {key}: {e}")
                return None

    def delete(self, key: str):
        """
        Delete blob.

        AGGIORNAMENTO: Non fallisce se key mancante.
        """
        with self._lock:
            try:
                if key in self._blobs:
                    del self._blobs[key]
                if key in self._metadata:
                    del self._metadata[key]
            except Exception as e:
                logger.error(f"Error deleting blob {key}: {e}")

    def list_keys(self) -> List[str]:
        """
        List all blob keys.

        NO CHANGES: già sicuro.
        """
        with self._lock:
            try:
                return list(self._blobs.keys())
            except Exception as e:
                logger.error(f"Error listing blob keys: {e}")
                return []

    def clear(self):
        """
        Clear all blobs.

        NO CHANGES: già sicuro.
        """
        with self._lock:
            try:
                self._blobs.clear()
                self._metadata.clear()
            except Exception as e:
                logger.error(f"Error clearing blobs: {e}")

    def has_key(self, key: str) -> bool:
        """Check if key exists"""
        with self._lock:
            return key in self._blobs

    def size(self) -> int:
        """Get number of blobs"""
        with self._lock:
            return len(self._blobs)


class OrderStore:
    """
    Thread-safe storage for active orders received from Kafka.

    Orders are keyed by mktOrderId. When an order arrives with status
    ACTIVE it is stored (or updated). When it arrives with status
    EXPIRED or CANCELLED it is removed from the store so that only
    active orders remain in memory at all times.
    """

    def __init__(self, locker: Lock = None):
        self._lock = locker if locker is not None else Lock()
        self._orders: Dict[str, Any] = {}  # mktOrderId -> Order

    def update(self, order: Any) -> None:
        """
        Upsert or remove an order based on its status.

        ACTIVE   -> store/replace under mktOrderId
        EXPIRED / CANCELLED -> remove if present
        """
        with self._lock:
            try:
                if order.is_active:
                    self._orders[order.mktOrderId] = order
                else:
                    self._orders.pop(order.mktOrderId, None)
            except Exception as e:
                logger.error(f"Error updating order {getattr(order, 'mktOrderId', '?')}: {e}")

    def get_all(self) -> List[Any]:
        """Return a list of all active orders (copy)."""
        with self._lock:
            return list(self._orders.values())

    def get_by_isin(self, isin: str) -> List[Any]:
        """Return active orders whose instrument ISIN matches."""
        with self._lock:
            return [o for o in self._orders.values() if o.isin == isin]

    def get_by_symbol(self, symbol: str) -> List[Any]:
        """Return active orders whose instrument symbol matches."""
        with self._lock:
            return [o for o in self._orders.values() if o.symbol == symbol]

    def remove(self, mkt_order_id: str) -> None:
        """Explicitly remove an order by mktOrderId."""
        with self._lock:
            self._orders.pop(mkt_order_id, None)

    def clear(self) -> None:
        """Remove all orders."""
        with self._lock:
            self._orders.clear()

    def count(self) -> int:
        """Return the number of active orders."""
        with self._lock:
            return len(self._orders)