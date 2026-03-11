"""
HistoricalDataFeed
==================
Carica dati storici di mercato da dizionari di DataFrame e li riproduce
come stream di eventi cronologici.

Formato input atteso
--------------------
    data = {
        "IE00B4L5Y983": pd.DataFrame(
            {"BID": [...], "ASK": [...]},
            index=pd.DatetimeIndex([...])  # o colonna "timestamp"
        ),
        "LU0048584102": pd.read_parquet("fund.parquet"),
    }

Ogni DataFrame deve avere un DatetimeIndex (o una colonna "timestamp").
Le colonne sono i campi di mercato (BID, ASK, MID, ...).

Utilizzo
--------
    feed = HistoricalDataFeed(data=data, fields=["BID", "ASK"])

    for ts, ticker, data_dict in feed.events():
        # ts       : datetime – timestamp simulato
        # ticker   : str      – identificativo strumento
        # data_dict: dict     – {field: value} con valori non-NaN
        rtdata.update(ticker, data_dict)
"""

from __future__ import annotations

import heapq
import itertools
import logging
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalDataFeed:
    """
    Fonde N stream di dati per-ticker in un unico stream ordinato per timestamp.

    Usa un min-heap per l'efficienza: O(T · log K) dove T = tick totali, K = ticker.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        fields: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            data:   Mappa ticker -> DataFrame con DatetimeIndex e colonne field.
            fields: Subset di colonne da estrarre. Default: tutte le colonne.

        Raises:
            ValueError: Se un DataFrame non ha un DatetimeIndex né una colonna "timestamp".
        """
        self.fields = fields
        self._data: Dict[str, pd.DataFrame] = self._validate_and_normalize(data, fields)
        self.tickers: List[str] = list(self._data.keys())

    # ------------------------------------------------------------------
    # Proprietà
    # ------------------------------------------------------------------

    @property
    def start_time(self) -> Optional[datetime]:
        """Timestamp del primo evento disponibile (su tutti i ticker)."""
        times = [df.index[0] for df in self._data.values() if len(df) > 0]
        return min(times).to_pydatetime() if times else None

    @property
    def end_time(self) -> Optional[datetime]:
        """Timestamp dell'ultimo evento disponibile (su tutti i ticker)."""
        times = [df.index[-1] for df in self._data.values() if len(df) > 0]
        return max(times).to_pydatetime() if times else None

    @property
    def total_ticks(self) -> int:
        """Numero totale di righe (tick) su tutti i ticker."""
        return sum(len(df) for df in self._data.values())

    def __len__(self) -> int:
        return self.total_ticks

    # ------------------------------------------------------------------
    # Stream di eventi
    # ------------------------------------------------------------------

    def events(self) -> Iterator[Tuple[datetime, str, Dict[str, float]]]:
        """
        Genera eventi in ordine cronologico.

        Yields:
            (timestamp, ticker, data_dict) dove data_dict contiene solo
            i campi con valore non-NaN della riga corrente.
        """
        heap: list = []
        counter = itertools.count()
        iterators: Dict[str, Iterator] = {}

        for ticker, df in self._data.items():
            it = iter(df.iterrows())
            iterators[ticker] = it
            try:
                ts, row = next(it)
                heapq.heappush(heap, (ts.to_pydatetime(), next(counter), ticker, row))
            except StopIteration:
                pass

        while heap:
            ts, _, ticker, row = heapq.heappop(heap)
            data_dict = {k: v for k, v in row.items() if pd.notna(v)}
            if data_dict:
                yield ts, ticker, data_dict

            try:
                ts2, row2 = next(iterators[ticker])
                heapq.heappush(
                    heap, (ts2.to_pydatetime(), next(counter), ticker, row2)
                )
            except StopIteration:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_normalize(
        data: Dict[str, pd.DataFrame],
        fields: Optional[List[str]],
    ) -> Dict[str, pd.DataFrame]:
        normalized: Dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            df = df.copy()

            # Normalizza indice a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                else:
                    raise ValueError(
                        f"Il DataFrame per '{ticker}' deve avere un DatetimeIndex "
                        f"oppure una colonna 'timestamp'. Tipo indice: {type(df.index)}"
                    )
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Filtra le colonne ai fields richiesti
            if fields:
                available = [f for f in fields if f in df.columns]
                missing = [f for f in fields if f not in df.columns]
                if missing:
                    logger.warning(
                        f"Ticker '{ticker}': colonne mancanti {missing}. "
                        f"Disponibili: {list(df.columns)}"
                    )
                df = df[available]

            # Rimuovi righe completamente vuote
            df = df.dropna(how="all")

            if len(df) == 0:
                logger.warning(f"Ticker '{ticker}': DataFrame vuoto dopo normalizzazione.")

            normalized[ticker] = df

        return normalized
