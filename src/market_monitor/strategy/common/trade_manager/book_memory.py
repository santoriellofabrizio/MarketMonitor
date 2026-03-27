from collections import deque
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass


class FairvaluePrice:
    """
    Rappresenta il prezzo mid di un singolo strumento finanziario, con supporto
    a tre modalità di indicizzazione: scalare, isin valuta, isin mercato.

    Il tipo di chiave viene fissato alla costruzione tramite i classmethod factory
    e non può essere modificato in seguito. Questo garantisce che ogni istanza abbia
    una semantica chiara e prevedibile.

    Attributes:
        isin (str): Identificativo dello strumento (tipicamente ISIN, ma può essere
            qualsiasi chiave usata nel sistema).

    Examples:
        >>> # Prezzo unico, indipendente da valuta/mercato
        >>> p = FairvaluePrice.scalar("IT0001234567", 102.3)
        >>> p.get()
        102.3

        >>> # Prezzo diverso per valuta
        >>> p = FairvaluePrice.by_currency("IE00B4L5Y983", {"EUR": 98.5, "USD": 99.1})
        >>> p.get(currency="EUR")
        98.5

        >>> # Prezzo diverso per mercato
        >>> p = FairvaluePrice.by_market("IE00B4L5Y983", {"MIL": 98.4, "LSE": 98.6})
        >>> p.get(market="LSE")
        98.6
    """

    __slots__ = ("isin", "_prices", "_key_type")

    def __init__(self, isin: str, prices: dict, key_type: str):
        """
        Costruttore interno. Preferire i classmethod factory:
        :meth:`scalar`, :meth:`by_currency`, :meth:`by_market`.

        Args:
            isin: Identificativo dello strumento.
            prices: Dizionario interno dei prezzi. Per ``scalar`` usa la chiave
                riservata ``"_scalar"``.
            key_type: Uno tra ``"scalar"``, ``"currency"``, ``"market"``.
        """
        self.isin = isin
        self._prices = prices
        self._key_type = key_type

    @classmethod
    def scalar(cls, isin: str | Any, value: float) -> "FairvaluePrice":
        """
        Crea un MidPrice con un singolo valore numerico, indipendente da
        valuta o mercato.

        Args:
            isin: Identificativo dello strumento.
            value: Prezzo mid.

        Returns:
            Istanza MidPrice di tipo ``scalar``.
        """
        return cls(isin=isin, prices={"_scalar": value}, key_type="scalar")

    @classmethod
    def by_currency(cls, isin: str, prices: dict[str, float]) -> "FairvaluePrice":
        """
        Crea un MidPrice con prezzi differenziati per valuta.

        Utile quando lo stesso strumento quota prezzi diversi in EUR, USD, ecc.

        Args:
            isin: Identificativo dello strumento.
            prices: Mappa ``{currency_code: mid_price}``.
                Esempio: ``{"EUR": 98.5, "USD": 99.1}``.

        Returns:
            Istanza MidPrice di tipo ``currency``.
        """
        return cls(isin=isin, prices=prices, key_type="currency")

    @classmethod
    def by_market(cls, isin: str, prices: dict[str, float]) -> "FairvaluePrice":
        """
        Crea un MidPrice con prezzi differenziati per mercato/sede di negoziazione.

        Utile quando lo stesso strumento quota su più exchange con prezzi distinti.

        Args:
            isin: Identificativo dello strumento.
            prices: Mappa ``{market_id: mid_price}``.
                Esempio: ``{"XPAR": 98.4, "XAMS": 98.6}``.

        Returns:
            Istanza MidPrice di tipo ``market``.
        """
        return cls(isin=isin, prices=prices, key_type="market")

    def get(self, currency: str | None = None, market: str | None = None) -> float | None:
        """
        Restituisce il prezzo mid per la chiave specificata.

        Il parametro da passare dipende dal tipo dell'istanza:

        - ``scalar``: nessun argomento necessario.
        - ``currency``: passare ``currency``.
        - ``market``: passare ``market``.

        Args:
            currency: Codice valuta (es. ``"EUR"``). Obbligatorio per istanze
                ``by_currency``, ignorato altrimenti.
            market: Identificativo mercato (es. ``"MIL"``). Obbligatorio per
                istanze ``by_market``, ignorato altrimenti.

        Returns:
            Il prezzo mid come ``float``, oppure ``None`` se la chiave non
            è presente nel dizionario interno.

        Raises:
            ValueError: Se l'istanza è ``by_currency`` e ``currency`` è ``None``,
                oppure se è ``by_market`` e ``market`` è ``None``.
        """
        if self._key_type == "scalar":
            return self._prices.get("_scalar")
        elif self._key_type == "currency":
            if currency is None:
                raise ValueError(f"[{self.isin}] key_type='currency' ma currency=None")
            return self._prices.get(currency)
        elif self._key_type == "market":
            if market is None:
                raise ValueError(f"[{self.isin}] key_type='market' ma market=None")
            return self._prices.get(market)

    @property
    def prices(self) -> dict:
        """
        Accesso in sola lettura al dizionario interno dei prezzi.

        Per istanze ``scalar`` contiene ``{"_scalar": value}``.
        Per ``currency`` e ``market`` contiene la mappa originale passata al factory.
        """
        return self._prices


# Alias chiaro per uno snapshot completo
BookSnapshot = dict[str, FairvaluePrice]  # isin -> MidPrice
"""
Tipo alias per uno snapshot istantaneo del book.

Mappa ogni ISIN (o identificativo strumento) alla corrispondente istanza
:class:`MidPrice`. Viene prodotto dalla strategia/feed e consumato da
:class:`BookStorage`.
"""


class BookStorage:
    """
    Storage circolare per snapshot temporali del book di prezzi mid.

    Ogni snapshot è un :data:`BookSnapshot` (``dict[isin, MidPrice]``).
    Gli snapshot vengono mantenuti in una ``deque`` a lunghezza fissa: quando
    viene raggiunto ``maxlen``, lo snapshot più vecchio viene scartato
    automaticamente.

    **Nessuna dipendenza da pandas**: non usa ``pd.Series``, ``MultiIndex``
    o ``index.name``. Tutto è basato su dizionari Python standard.

    Args:
        maxlen: Numero massimo di snapshot da conservare in memoria.
            Default ``3``.

    Examples:
        Caso d'uso tipico in una strategia:

        >>> storage = BookStorage(maxlen=5)
        >>>
        >>> # Ad ogni tick del feed:
        >>> snapshot = {
        ...     "IE00B4L5Y983": FairvaluePrice.by_currency("IE00B4L5Y983", {"EUR": 98.5, "USD": 99.1}),
        ...     "IT0001234567": FairvaluePrice.scalar("IT0001234567", 102.3),
        ... }
        >>> storage.append(snapshot)
        >>>
        >>> # Lettura del prezzo più recente:
        >>> storage.get_mid("IE00B4L5Y983", currency="EUR")
        98.5
        >>> storage.get_mid("IT0001234567")
        102.3
        >>>
        >>> # Confronto con il prezzo precedente:
        >>> storage.get_mid("IT0001234567", old=True)   # snapshot più vecchio
        >>>
        >>> # Accesso per timestamp:
        >>> ts = datetime(2024, 1, 15, 10, 30)
        >>> storage.get_last_before(ts)   # (datetime, BookSnapshot) | None
    """

    def __init__(self, maxlen: int = 20):
        self._storage: deque[tuple[datetime, BookSnapshot]] = deque(maxlen=maxlen)

    def append(self, snapshot: BookSnapshot, time_snapshot: datetime | None = None) -> None:
        """
        Aggiunge uno snapshot al buffer.

        Se non viene fornito un timestamp, viene usato ``datetime.now()``.
        Quando il buffer è pieno (``len == maxlen``), lo snapshot più vecchio
        viene rimosso automaticamente dalla deque.

        Args:
            snapshot: Dizionario ``{isin: MidPrice}`` da memorizzare.
            time_snapshot: Timestamp associato allo snapshot. Se ``None``,
                viene usato l'orario corrente.
        """
        ts = time_snapshot or datetime.now()
        self._storage.append((ts, snapshot))

    def get_mid(
        self,
        isin: str,
        currency: str | None = None,
        market: str | None = None,
        old: bool = False,
    ) -> float | None:
        """
        Restituisce il prezzo mid di uno strumento dallo snapshot più recente
        (o dal più vecchio se ``old=True``).

        Args:
            isin: Identificativo dello strumento da cercare.
            currency: Codice valuta, richiesto per istanze :class:`MidPrice`
                di tipo ``by_currency``.
            market: Identificativo mercato, richiesto per istanze :class:`MidPrice`
                di tipo ``by_market``.
            old: Se ``True``, legge dallo snapshot più vecchio presente nel buffer
                invece che dal più recente. Utile per confronti tick-su-tick.
                Default ``False``.

        Returns:
            Il prezzo mid come ``float``, oppure ``None`` se il buffer è vuoto
            o l'ISIN non è presente nello snapshot.
        """
        if not self._storage:
            return None
        _, snapshot = self._storage[0 if old else -1]
        mid = snapshot.get(isin)
        return mid.get(currency=currency, market=market) if mid else None

    def get_last_before(self, timestamp: datetime) -> Optional[tuple[datetime, BookSnapshot]]:
        """
        Restituisce lo snapshot più recente con timestamp ``<= timestamp``.

        Scorre il buffer dall'ultimo al primo e restituisce il primo che
        soddisfa la condizione. Accetta anche timestamp in nanosecondi (int),
        che vengono convertiti automaticamente.

        Args:
            timestamp: Limite superiore (inclusivo) per la ricerca.
                Può essere un ``datetime`` o un intero in nanosecondi (es.
                timestamp Unix nanosecondo da feed di mercato).

        Returns:
            Tupla ``(datetime, BookSnapshot)`` dello snapshot trovato,
            oppure ``None`` se il buffer è vuoto o tutti gli snapshot sono
            successivi a ``timestamp``.
        """
        if isinstance(timestamp, int):
            timestamp = datetime.fromtimestamp(timestamp / 1_000_000_000)
        for ts, snapshot in reversed(self._storage):
            if ts <= timestamp:
                return ts, snapshot
        return None

    def get_first_after(self, timestamp: datetime) -> Optional[tuple[datetime, BookSnapshot]]:
        """
        Restituisce lo snapshot meno recente con timestamp ``>= timestamp``.

        Scorre il buffer dal primo all'ultimo e restituisce il primo che
        soddisfa la condizione.

        Args:
            timestamp: Limite inferiore (inclusivo) per la ricerca.

        Returns:
            Tupla ``(datetime, BookSnapshot)`` dello snapshot trovato,
            oppure ``None`` se il buffer è vuoto o tutti gli snapshot
            precedono ``timestamp``.
        """
        for ts, snapshot in self._storage:
            if ts >= timestamp:
                return ts, snapshot
        return None

    def get_age_seconds(self, old: bool = False) -> float | None:
        """
        Restituisce l'età in secondi dello snapshot più recente (o del più vecchio).

        Misura il tempo trascorso tra il timestamp dello snapshot e ``datetime.now()``.
        Utile per monitorare lo staleness del book (es. scartare prezzi troppo vecchi).

        Args:
            old: Se ``True``, restituisce l'età dello snapshot più vecchio
                nel buffer. Default ``False`` (snapshot più recente).

        Returns:
            Età in secondi come ``float``, oppure ``None`` se il buffer è vuoto.
        """
        if not self._storage:
            return None
        ts, _ = self._storage[0 if old else -1]
        return (datetime.now() - ts).total_seconds()

    def __len__(self) -> int:
        """Numero di snapshot attualmente nel buffer."""
        return len(self._storage)

    def __bool__(self) -> bool:
        """``True`` se il buffer contiene almeno uno snapshot."""
        return bool(self._storage)

    def __getitem__(self, index: int) -> tuple[datetime, BookSnapshot]:
        """
        Accesso diretto a uno snapshot per indice intero.

        Segue la semantica della ``deque``: indice ``0`` è il più vecchio,
        indice ``-1`` è il più recente.

        Args:
            index: Indice intero (positivo o negativo).

        Returns:
            Tupla ``(datetime, BookSnapshot)``.

        Raises:
            IndexError: Se l'indice è fuori range.
        """
        return self._storage[index]