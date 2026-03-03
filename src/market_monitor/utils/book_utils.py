import math
import time
import pandas as pd
from typing import Protocol, runtime_checkable


@runtime_checkable
class BookFilter(Protocol):
    def update(self, book_raw: pd.DataFrame) -> None: ...
    def get_valid_book(self, book_raw: pd.DataFrame) -> pd.DataFrame: ...


class SpreadEWMA(BookFilter):
    """
    Usage:
        >>> ewma = SpreadEWMA(tau_seconds=600, max_multiplier=2.0)
        >>> ewma.update(book_raw)
        >>> valid_book = ewma.get_valid_book(book_raw)
    """

    def __init__(self, tau_seconds: float = 600, max_multiplier: float | dict[str, float] = 2.0, **kwargs):
        self.tau_seconds = tau_seconds
        self.max_multiplier = max_multiplier
        self._ewma: pd.Series | None = None
        self._last_update: float | None = None  # time.monotonic()

    def _get_max_multiplier(self, instrument_id: str) -> float:
        if isinstance(self.max_multiplier, dict):
            return self.max_multiplier.get(instrument_id, 2.0)
        return self.max_multiplier

    def _tau_weight(self) -> float:
        if self._last_update is None:
            return 0.0  # primo tick: peso 0 → EWMA = spread corrente
        dt = time.monotonic() - self._last_update
        return math.exp(-dt / self.tau_seconds)

    def update(self, book_raw: pd.DataFrame) -> None:
        if not {"BID", "ASK"}.issubset(book_raw.columns):
            raise ValueError(f"book_raw deve avere colonne BID e ASK, trovato: {list(book_raw.columns)}")
        if book_raw.empty:
            return

        spread = book_raw["ASK"] / book_raw["BID"] - 1
        alpha = self._tau_weight()

        if self._ewma is None:
            self._ewma = spread.copy()
        else:
            new_instruments = spread.index.difference(self._ewma.index)
            known_instruments = spread.index.intersection(self._ewma.index)

            self._ewma[known_instruments] = alpha * self._ewma[known_instruments] + (1 - alpha) * spread[
                known_instruments]
            if not new_instruments.empty:
                self._ewma = pd.concat([self._ewma, spread[new_instruments]])

        self._last_update = time.monotonic()

    def get_valid_book(self, book_raw: pd.DataFrame) -> pd.DataFrame:
        if not {"BID", "ASK"}.issubset(book_raw.columns):
            raise ValueError(f"book_raw deve avere colonne BID e ASK, trovato: {list(book_raw.columns)}")
        if self._ewma is None:
            return book_raw

        valid_ewma = self._ewma.reindex(book_raw.index)
        max_mult = pd.Series({i: self._get_max_multiplier(i) for i in book_raw.index})
        current_spread = book_raw["ASK"] / book_raw["BID"] - 1

        # strumenti senza EWMA (nuovi) passano sempre il filtro
        mask = current_spread <= max_mult * valid_ewma.fillna(float("inf"))

        _isin = "ETFP_IE000YZIVX22"
        if _isin in book_raw.index:
            _spread = current_spread.loc[_isin]
            _ewma = valid_ewma.loc[_isin]
            _mult = max_mult.loc[_isin]
            _pass = mask.loc[_isin]
            print(
                f"FILTER: CATHEM spread={_spread * 100:.2f}% | "
                f"ewma={_ewma * 100:.2f}% | "
                f"threshold={_mult * _ewma * 100:.2f}% ({_mult}x) | "
                f"{'PASS ✓' if _pass else 'FILTERED ✗'}"
            )

        return book_raw[mask]

class PriceEWMA(BookFilter):
    """
    Usage:
        >>> ewma = PriceEWMA(tau_seconds=600, max_ret=0.005)
        >>> ewma.update(raw)
        >>> valid = ewma.get_valid_book(raw)
    """

    def __init__(self, tau_seconds: float, max_ret: float | dict[str, float] = 0.005, **kwargs):
        self.tau_seconds = tau_seconds
        self.max_ret = max_ret
        self._ewma_bid: pd.Series | None = None
        self._ewma_ask: pd.Series | None = None
        self._last_update: float | None = None

    def _get_max_ret(self, instrument_id: str) -> float:
        if isinstance(self.max_ret, dict):
            return self.max_ret.get(instrument_id, 0.005)
        return self.max_ret

    def _tau_weight(self) -> float:
        if self._last_update is None:
            return 0.0
        dt = time.monotonic() - self._last_update
        return math.exp(-dt / self.tau_seconds)

    def update(self, book_raw: pd.DataFrame) -> None:
        if not {"BID", "ASK"}.issubset(book_raw.columns):
            raise ValueError(f"book_raw deve avere colonne BID e ASK, trovato: {list(book_raw.columns)}")

        alpha = self._tau_weight()

        if self._ewma_bid is None:
            self._ewma_bid = book_raw["BID"].copy()
            self._ewma_ask = book_raw["ASK"].copy()
        else:
            new_instruments = book_raw.index.difference(self._ewma_bid.index)
            known_instruments = book_raw.index.intersection(self._ewma_bid.index)

            self._ewma_bid[known_instruments] = alpha * self._ewma_bid[known_instruments] + (1 - alpha) * book_raw.loc[
                known_instruments, "BID"]
            self._ewma_ask[known_instruments] = alpha * self._ewma_ask[known_instruments] + (1 - alpha) * book_raw.loc[
                known_instruments, "ASK"]

            if not new_instruments.empty:
                self._ewma_bid = pd.concat([self._ewma_bid, book_raw.loc[new_instruments, "BID"]])
                self._ewma_ask = pd.concat([self._ewma_ask, book_raw.loc[new_instruments, "ASK"]])

        self._last_update = time.monotonic()

    def get_valid_book(self, book_raw: pd.DataFrame) -> pd.DataFrame:
        if not {"BID", "ASK"}.issubset(book_raw.columns):
            raise ValueError(f"book_raw deve avere colonne BID e ASK, trovato: {list(book_raw.columns)}")
        if self._ewma_bid is None:
            return book_raw

        ewma_bid = self._ewma_bid.reindex(book_raw.index).fillna(float("inf"))
        ewma_ask = self._ewma_ask.reindex(book_raw.index).fillna(float("inf"))
        max_ret_series = pd.Series({i: self._get_max_ret(i) for i in book_raw.index})

        bid_ret = (book_raw["BID"] / ewma_bid - 1).abs()
        ask_ret = (book_raw["ASK"] / ewma_ask - 1).abs()
        mask = (bid_ret <= max_ret_series) & (ask_ret <= max_ret_series)
        return book_raw[mask]