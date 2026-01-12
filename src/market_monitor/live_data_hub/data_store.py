"""
Store classes refactored per robustezza.

AGGIORNAMENTO: Gestione sicura di indici/colonne mancanti.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketStore:
    """
    DataFrame-based storage for market data (prices from Bloomberg/RedisPublisher).
    Thread-safe with minimal pandas overhead.

    AGGIORNAMENTO: Metodi robusti che gestiscono indici/colonne mancanti.
    """

    def __init__(self, locker: Lock, fields: List[str]):
        self._lock = locker
        self._fields = fields
        self._market_data: pd.DataFrame = pd.DataFrame()
        self._last_update: Dict[str, datetime] = {}

    def initialize(self, securities: List[str]):
        """Initialize market DataFrame for securities"""
        with self._lock:
            self._market_data = pd.DataFrame(
                index=securities,
                columns=self._fields,
                dtype=float
            )
            # EUR always at 1.0
            if "EUR" not in self._market_data.index:
                self._market_data.loc["EUR"] = [1.0] * len(self._fields)

    def update(self, ticker: str, data: Dict[str, float]):
        """
        Update market data for ticker.

        AGGIORNAMENTO: Crea automaticamente ticker/colonne se mancanti.
        """

        with self._lock:
            try:
                #  Se DataFrame è completamente vuoto, inizializzalo con i campi necessari
                if self._market_data.empty:
                    # Usa i campi da data + fields predefiniti
                    all_fields = list(set(self._fields + list(data.keys())))
                    self._market_data = pd.DataFrame(columns=all_fields, dtype=float)

                # Aggiungi colonne mancanti PRIMA di aggiungere il ticker
                for field in data.keys():
                    if field not in self._market_data.columns:
                        self._market_data[field] = np.nan

                #  Ora possiamo aggiungere il ticker (le colonne esistono)
                if ticker not in self._market_data.index:
                    self._market_data.loc[ticker] = np.nan

                # Update valori
                for field, value in data.items():
                    self._market_data.at[ticker, field] = value
                self._last_update[ticker] = datetime.now()
                return

            except Exception as e:
                logger.error(f"Error updating market data for {ticker}: {e}")
                return False, list(data.keys())

    def get_data(self, tickers: Optional[List[str]] = None,
                 fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get market data (thread-safe copy).

        AGGIORNAMENTO: Gestisce indici/colonne mancanti.
        """
        with self._lock:
            #  DataFrame vuoto
            if self._market_data.empty:
                return pd.DataFrame()

            try:
                if tickers is None and fields is None:
                    return self._market_data.copy()

                elif tickers is None:
                    # Filtra solo colonne esistenti
                    existing_fields = [f for f in fields if f in self._market_data.columns]
                    if not existing_fields:
                        return pd.DataFrame()
                    return self._market_data[existing_fields].copy()

                elif fields is None:
                    #  Filtra solo ticker esistenti
                    existing_tickers = [t for t in tickers if t in self._market_data.index]
                    if not existing_tickers:
                        return pd.DataFrame()
                    return self._market_data.loc[existing_tickers].copy()

                else:
                    #  Filtra entrambi
                    existing_tickers = [t for t in tickers if t in self._market_data.index]
                    existing_fields = [f for f in fields if f in self._market_data.columns]
                    if not existing_tickers or not existing_fields:
                        return pd.DataFrame()
                    return self._market_data.loc[existing_tickers, existing_fields].copy()

            except Exception as e:
                logger.error(f"Error getting market data: {e}")
                return pd.DataFrame()

    def get_field(self, field: str, tickers: Optional[List[str]] = None) -> pd.Series:
        """
        Get specific field for tickers.

        AGGIORNAMENTO: Ritorna Series vuota se campo/ticker mancante.
        """
        with self._lock:
            # Controlli esistenza
            if self._market_data.empty or field not in self._market_data.columns:
                return pd.Series(dtype=float)

            try:
                if tickers is None:
                    return self._market_data[field].copy()
                else:
                    # Filtra ticker esistenti
                    existing_tickers = [t for t in tickers if t in self._market_data.index]
                    if not existing_tickers:
                        return pd.Series(dtype=float)
                    return self._market_data.loc[existing_tickers, field].copy()

            except Exception as e:
                logger.error(f"Error getting field {field}: {e}")
                return pd.Series(dtype=float)

    def get_mid(self, mid_fields: List[str],
                tickers: Optional[List[str]] = None) -> pd.Series:
        """
        Calculate mid prices (optimized with numpy).

        AGGIORNAMENTO: Gestisce campi mancanti.
        """
        with self._lock:
            # Controlli esistenza
            if self._market_data.empty:
                return pd.Series(dtype=float)

            # Filtra solo campi esistenti
            existing_fields = [f for f in mid_fields if f in self._market_data.columns]
            if not existing_fields:
                return pd.Series(dtype=float)

            try:
                if tickers is None:
                    data = self._market_data[existing_fields]
                else:
                    # Filtra ticker esistenti
                    existing_tickers = [t for t in tickers if t in self._market_data.index]
                    if not existing_tickers:
                        return pd.Series(dtype=float)
                    data = self._market_data.loc[existing_tickers, existing_fields]

                return pd.Series(data.values.mean(axis=1), index=data.index)

            except Exception as e:
                logger.error(f"Error calculating mid prices: {e}")
                return pd.Series(dtype=float)

    def get_mid_dict(self, mid_fields: List[str],
                     tickers: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get mid prices as dict (faster for lookups).

        AGGIORNAMENTO: Ritorna dict vuoto se campi mancanti.
        """
        with self._lock:
            # Controlli esistenza
            if self._market_data.empty:
                return {}

            # Filtra solo campi esistenti
            existing_fields = [f for f in mid_fields if f in self._market_data.columns]
            if not existing_fields:
                return {}

            try:
                if tickers is None:
                    data = self._market_data[existing_fields]
                else:
                    # Filtra ticker esistenti
                    existing_tickers = [t for t in tickers if t in self._market_data.index]
                    if not existing_tickers:
                        return {}
                    data = self._market_data.loc[existing_tickers, existing_fields]

                means = data.values.mean(axis=1)
                return {ticker: means[i] for i, ticker in enumerate(data.index)}

            except Exception as e:
                logger.error(f"Error getting mid dict: {e}")
                return {}

    def get_last_update(self, ticker: str) -> Optional[datetime]:
        """Get last update time for ticker"""
        with self._lock:
            return self._last_update.get(ticker)

    def get_securities(self) -> List[str]:
        """Get list of securities"""
        with self._lock:
            return list(self._market_data.index) if not self._market_data.empty else []

    def set_dataframe(self, df: pd.DataFrame):
        """Replace entire DataFrame"""
        with self._lock:
            self._market_data = df.copy() if not df.empty else pd.DataFrame()

    def has_ticker(self, ticker: str) -> bool:
        """Check if ticker exists"""
        with self._lock:
            return ticker in self._market_data.index if not self._market_data.empty else False

    def has_field(self, field: str) -> bool:
        """Check if field exists"""
        with self._lock:
            return field in self._market_data.columns if not self._market_data.empty else False


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

    def __init__(self, locker: Lock):
        self._lock = locker
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

    def __init__(self, locker: Lock, default_maxlen: int = 1000):
        self._lock = locker
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

    def __init__(self, locker: Lock):
        self._lock = locker
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