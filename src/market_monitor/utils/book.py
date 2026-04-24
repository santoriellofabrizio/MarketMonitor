"""
CompositeBook: vectorised live book aggregator with pluggable filters.

Terminology
-----------
sub_id        : subscription key used with market_data (e.g. "IM:IE00B4L5Y983")
instrument_id : canonical security key — required at registration
market        : venue / segment — required at registration
currency      : ISO currency code — required at registration
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from market_monitor.strategy.common.trade_manager.book_memory import (
    FairvaluePrice, BookSnapshot,
)
from market_monitor.utils.book_utils import BookFilter

# Internal column names attached to _clean_book by update()
_COL_INSTR = "_INSTR_ID"
_COL_MKT   = "_MARKET"
_COL_CCY   = "_CURRENCY"

_DIM_TO_COL: dict[str, str] = {"MARKET": _COL_MKT, "CURRENCY": _COL_CCY}
_ALL_DIMS: list[str] = list(_DIM_TO_COL.keys())  # canonical order preserved in _by_keep


# ── Aggregation presets ───────────────────────────────────────────────────────
# Pass these (or any lambda) as agg_function to BookQuery.get_field().
# Single-field queries receive a pd.Series; multi-field queries receive a
# pd.DataFrame with one row per subscription in the group.

def best_bid_ask(df: pd.DataFrame) -> float:
    """Best composite mid: max(BID) and min(ASK) across subscriptions."""
    return (df["BID"].max() + df["ASK"].min()) / 2


def mean_bid_ask(df: pd.DataFrame) -> float:
    """Mid from the mean bid and the mean ask across subscriptions."""
    return (df["BID"].mean() + df["ASK"].mean()) / 2


def mean_mids(df: pd.DataFrame) -> float:
    """Mean of per-subscription (BID + ASK) / 2."""
    return ((df["BID"] + df["ASK"]) / 2).mean()


# ── BookResult ────────────────────────────────────────────────────────────────

class BookResult:
    """
    Immutable result of ``CompositeBook.agg(by=...).get_field(...)``.

    ``_by`` holds the dimensions that were **kept** (not collapsed), which
    determines the tuple-key structure of the result.

    Key shapes:
        agg(by=["MARKET","CURRENCY"]) → ("IE00B",)            — fully collapsed
        agg(by=["MARKET"])            → ("IE00B", "EUR")       — currency kept
        agg(by=["CURRENCY"])          → ("IE00B", "IM")        — market kept
        agg(by=[])                    → ("IE00B", "IM", "EUR") — nothing collapsed
    """

    def __init__(self, data: pd.Series, by_keep: list[str]) -> None:
        self._data   = data
        self._by     = by_keep  # e.g. [] or ["CURRENCY"] or ["MARKET", "CURRENCY"]

    # ── Conversion ────────────────────────────────────────────────────────────

    def as_dict(self) -> dict[tuple, float]:
        """Raw ``{tuple_key: price}`` dict."""
        return self._data.to_dict()

    def as_series(self) -> pd.Series:
        """
        Flat ``pd.Series`` (instr_id index) when no dims are kept;
        ``MultiIndex`` Series (instr_id, kept_dim, …) otherwise.
        """
        if self._data.empty:
            return pd.Series(dtype=float)
        if not self._by:
            return pd.Series(
                self._data.values,
                index=[k[0] for k in self._data.index],
                dtype=float,
            )
        idx = pd.MultiIndex.from_tuples(
            self._data.index, names=["instr_id"] + self._by
        )
        return pd.Series(self._data.values, index=idx, dtype=float)

    def as_df(self) -> pd.DataFrame:
        """
        Single-column DataFrame when no dims are kept;
        pivoted ``(instr_id rows × last-dim columns)`` otherwise.
        """
        s = self.as_series()
        if not self._by:
            return s.to_frame(name="mid")
        return s.unstack(level=-1)

    def as_snapshot(self) -> BookSnapshot:
        """
        Convert to ``BookSnapshot`` for ``BookStorage.append()`` / TradeManager.

        Mapping:
          no kept dims        → ``FairvaluePrice.scalar``      per instr_id
          kept=["CURRENCY"]   → ``FairvaluePrice.by_currency`` per instr_id
          kept=["MARKET"]     → ``FairvaluePrice.by_market``   per instr_id
          kept both           → ``by_currency`` (CURRENCY wins)
        """
        snapshot: BookSnapshot = {}

        if not self._by:
            for (instr_id,), price in self._data.items():
                if not np.isnan(price):
                    snapshot[instr_id] = FairvaluePrice.scalar(instr_id, price)

        elif self._by == ["CURRENCY"]:
            grouped: dict[str, dict] = defaultdict(dict)
            for (instr_id, ccy), price in self._data.items():
                if not np.isnan(price):
                    grouped[instr_id][ccy] = price
            for instr_id, prices in grouped.items():
                snapshot[instr_id] = FairvaluePrice.by_currency(instr_id, prices)

        elif self._by == ["MARKET"]:
            grouped = defaultdict(dict)
            for (instr_id, mkt), price in self._data.items():
                if not np.isnan(price):
                    grouped[instr_id][mkt] = price
            for instr_id, prices in grouped.items():
                snapshot[instr_id] = FairvaluePrice.by_market(instr_id, prices)

        else:
            # Multi-dim: CURRENCY wins for TradeManager compat
            if "CURRENCY" in self._by:
                ccy_pos = self._by.index("CURRENCY") + 1
                grouped = defaultdict(dict)
                for key, price in self._data.items():
                    if not np.isnan(price):
                        grouped[key[0]][key[ccy_pos]] = price
                for instr_id, prices in grouped.items():
                    snapshot[instr_id] = FairvaluePrice.by_currency(instr_id, prices)
            else:
                mkt_pos = self._by.index("MARKET") + 1
                grouped = defaultdict(dict)
                for key, price in self._data.items():
                    if not np.isnan(price):
                        grouped[key[0]][key[mkt_pos]] = price
                for instr_id, prices in grouped.items():
                    snapshot[instr_id] = FairvaluePrice.by_market(instr_id, prices)

        return snapshot


# ── BookQuery ─────────────────────────────────────────────────────────────────

class BookQuery:
    """
    Pending query produced by ``CompositeBook.agg(by=[...])``.

    ``by`` lists the dimensions to **collapse** (aggregate out); dimensions
    NOT listed are kept as extra tuple levels in the result key.

    Execute with ``.get_field(fields, agg_function)`` → :class:`BookResult`.
    """

    def __init__(self, book: "CompositeBook", by: list[str]) -> None:
        unknown = set(by) - _DIM_TO_COL.keys()
        if unknown:
            raise ValueError(
                f"Unknown dimension(s) {unknown}. Valid: {list(_DIM_TO_COL)}"
            )
        self._book     = book
        self._by_keep  = [d for d in _ALL_DIMS if d not in by]

    # ── Core query ────────────────────────────────────────────────────────────

    def get_field(
        self,
        fields: str | list[str],
        agg_function: Callable | str,
        securities: list[str] | None = None,
        fx_rate: dict[str, float] | pd.Series | None = None,
    ) -> BookResult:
        """
        Aggregate ``fields`` within each group using ``agg_function``.

        Parameters
        ----------
        fields :
            A single column name (``str``) or a list of column names.
            The clean book exposes at least ``BID``, ``ASK`` and any extra
            fields from the raw market-data snapshot.
        agg_function :
            How to reduce each group to a scalar.

            * **Single field** — called as ``Series.agg(agg_function)``:
              pass ``"max"``, ``"min"``, ``np.mean``, or any callable that
              takes a ``pd.Series`` and returns a scalar.

            * **Multiple fields** — called via ``DataFrame.groupby.apply``:
              pass a callable ``f(df: pd.DataFrame) -> float`` that receives
              one row per subscription in the group.
              Module-level presets: :func:`best_bid_ask`, :func:`mean_bid_ask`,
              :func:`mean_mids`.

        securities :
            Optional list of instrument_ids to restrict the result.
        fx_rate :
            ``{currency: rate}`` mapping applied to BID/ASK **after** filters
            and **before** aggregation (converts prices to a common currency).
            e.g. ``{"EUR": 1.0, "USD": 0.92, "GBP": 1.16}``

        Returns
        -------
        BookResult
            Indexed by ``(instr_id,)`` when all dims are collapsed, or by
            ``(instr_id, kept_dim, …)`` otherwise.

        Examples
        --------
        >>> # best composite mid, collapse everything → flat Series
        >>> result = book.agg(by=["MARKET","CURRENCY"]).get_field(
        ...     ["BID","ASK"], best_bid_ask
        ... )
        >>> result.as_series()   # index = instr_id

        >>> # max bid per (ISIN, market)
        >>> result = book.agg(by=["CURRENCY"]).get_field("BID", "max")

        >>> # custom lambda: VWAP-style mid
        >>> result = book.agg(by=["MARKET","CURRENCY"]).get_field(
        ...     ["BID","ASK"], lambda df: (df["BID"] + df["ASK"]).mean() / 2
        ... )

        >>> # fallback to LAST_PRICE
        >>> mid = (
        ...     book.agg(by=["MARKET","CURRENCY"])
        ...         .get_field(["BID","ASK"], best_bid_ask)
        ...         .as_series()
        ...         .combine_first(last_book["LAST_PRICE"])
        ... )
        """
        book = self._book._clean_book
        if book is None or book.empty:
            return BookResult(pd.Series(dtype=float), self._by_keep)

        if securities is not None:
            book = book[book[_COL_INSTR].isin(securities)]
        if book.empty:
            return BookResult(pd.Series(dtype=float), self._by_keep)

        # FX conversion to a common currency (after filters, before aggregation)
        if fx_rate is not None:
            bid_ask = [c for c in ("BID", "ASK") if c in book.columns]
            if bid_ask:
                fx   = fx_rate if isinstance(fx_rate, dict) else fx_rate.to_dict()
                rates = book[_COL_CCY].map(fx).fillna(1.0)
                book  = book.copy()
                book[bid_ask] = book[bid_ask].multiply(rates, axis=0)

        # Build the groupby key — tuple of (instr_id [, kept_dim …])
        key_cols  = [_COL_INSTR] + [_DIM_TO_COL[d] for d in self._by_keep]
        by_series = book[key_cols].apply(tuple, axis=1)

        try:
            if isinstance(fields, str):
                result = book[fields].groupby(by_series).agg(agg_function)
            else:
                result = book[fields].groupby(by_series).apply(agg_function)
        except (KeyError, TypeError):
            result = pd.Series(dtype=float)

        return BookResult(result, self._by_keep)


# ── CompositeBook ─────────────────────────────────────────────────────────────

class CompositeBook:
    """
    Aggregates live market data from one or more subscriptions per instrument.

    ---

    USAGE GUIDE
    ===========

    1. Registration — instr_id, market and currency are all mandatory
    ----------------------------------------------------------------

        book = CompositeBook()

        book.register("IM:IE00B4L5Y983",
                      instr_id="IE00B4L5Y983", market="IM", currency="EUR")

        for mkt, ccy in [("IM", "EUR"), ("FP", "EUR"), ("NA", "USD")]:
            book.register(f"{mkt}:IE00B4L5Y983",
                          instr_id="IE00B4L5Y983", market=mkt, currency=ccy)

        book.register_from_instruments({inst.id: inst for inst in my_instruments})

    2. Filters
    ----------

        from market_monitor.utils.book_utils import SpreadEWMA, PriceEWMA

        book.add_filter(SpreadEWMA(tau_seconds=600, max_multiplier=2.0))
        book.add_filter(PriceEWMA(tau_seconds=300, max_ret=0.005),
                        securities=["IM:IE00B4L5Y983"])

    3. Hot path — update every tick
    --------------------------------

        raw = market_data.get_data_field(field=["BID", "ASK"])
        book.update(raw)

    4. Querying — agg / get_field / BookResult
    -------------------------------------------

        ``by`` = dimensions to COLLAPSE (aggregate out).

        # fully aggregate → flat Series indexed by instr_id
        result = book.agg(by=["MARKET","CURRENCY"]).get_field(
            ["BID","ASK"], best_bid_ask
        )

        # collapse only market → (instr_id, currency) MultiIndex
        result = book.agg(by=["MARKET"]).get_field(["BID","ASK"], best_bid_ask)

        # no collapse → (instr_id, market, currency) MultiIndex
        result = book.agg(by=[]).get_field(["BID","ASK"], best_bid_ask)

        # single field
        result = book.agg(by=["MARKET","CURRENCY"]).get_field("BID", "max")

        # custom lambda + FX conversion
        fx = {"EUR": 1.0, "USD": 0.92}
        result = (
            book.agg(by=["MARKET","CURRENCY"])
                .get_field(["BID","ASK"], best_bid_ask,
                           securities=my_isins, fx_rate=fx)
        )

    5. Consuming BookResult
    -----------------------

        result.as_series()          # flat or MultiIndex pd.Series
        result.as_df()              # DataFrame (pivoted when dims are kept)
        result.as_dict()            # {tuple_key: price}
        result.as_snapshot()        # BookSnapshot for BookStorage / TradeManager

        book_storage.append(result.as_snapshot())

        # combine with LAST_PRICE fallback
        mid = (
            book.agg(by=["MARKET","CURRENCY"])
                .get_field(["BID","ASK"], best_bid_ask)
                .as_series()
                .combine_first(last_book["LAST_PRICE"])
        )

    6. Per-instrument bid / ask
    ---------------------------

        bid, ask = book.get_bid_ask("IE00B4L5Y983")
    """

    def __init__(self) -> None:
        self._sub_to_instr:  dict[str, str]              = {}
        self._sub_metadata:  dict[str, tuple[str, str]]  = {}  # sub_id → (market, currency)
        self._filters:       list[tuple[BookFilter, frozenset[str] | None]] = []
        self._clean_book:    pd.DataFrame | None         = None
        self._bid:           pd.Series | None            = None
        self._ask:           pd.Series | None            = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        sub_id: str,
        instr_id: str,
        market: str = "GenericMarket",
        currency: str = "GenericMarket",
    ) -> "CompositeBook":
        """Register one subscription. All four arguments are required."""
        self._sub_to_instr[sub_id]  = instr_id
        self._sub_metadata[sub_id]  = (market, currency)
        return self

    def register_from_instruments(self, instruments: dict) -> "CompositeBook":
        """
        Bulk-register from a ``{sub_id: instr_obj}`` dict.

        ``instr_obj`` must expose ``.id``, ``.market`` and ``.currency``.
        """
        for sub_id, instr in instruments.items():
            self.register(sub_id=sub_id, instr_id=instr.id,
                          market=instr.market, currency=instr.currency)
        return self

    # ── Configuration ─────────────────────────────────────────────────────────

    def add_filter(
        self,
        filt: BookFilter,
        securities: Sequence[str] | None = None,
    ) -> "CompositeBook":
        """Attach a ``BookFilter``. ``securities=None`` → global; otherwise scoped."""
        self._filters.append(
            (filt, frozenset(securities) if securities is not None else None)
        )
        return self

    # ── Update (hot path) ─────────────────────────────────────────────────────

    def update(self, raw: pd.DataFrame) -> None:
        """
        Ingest a market-data snapshot, apply filters, and cache the result.

        ``raw``: DataFrame with index=sub_id and any field columns (BID, ASK, …).
        Zero values are treated as NaN.
        """
        self._clean_book = None
        self._bid = pd.Series(dtype=float)
        self._ask = pd.Series(dtype=float)

        if raw.empty:
            return

        book = raw.replace({0: np.nan}).dropna(how="all")
        if book.empty:
            return

        for filt, scope in self._filters:
            if not {"BID", "ASK"}.issubset(book.columns):
                continue
            if scope is None:
                filt.update(book)
                book = filt.get_valid_book(book)
            else:
                idx = book.index.intersection(scope)
                if idx.empty:
                    continue
                subset = book.loc[idx]
                filt.update(subset)
                valid = filt.get_valid_book(subset)
                book = pd.concat([book.loc[book.index.difference(scope)], valid])

        if book.empty:
            return

        known = book.index.intersection(self._sub_to_instr.keys())
        book  = book.reindex(known).copy()
        book[_COL_INSTR] = pd.Series(self._sub_to_instr).reindex(book.index)
        book[_COL_MKT]   = pd.Series({s: self._sub_metadata[s][0] for s in book.index})
        book[_COL_CCY]   = pd.Series({s: self._sub_metadata[s][1] for s in book.index})
        self._clean_book = book.dropna(subset=[_COL_INSTR])

        instr_by = self._clean_book[_COL_INSTR]
        if "BID" in self._clean_book.columns:
            self._bid = self._clean_book["BID"].groupby(instr_by).max()
        if "ASK" in self._clean_book.columns:
            self._ask = self._clean_book["ASK"].groupby(instr_by).min()

    # ── Query interface ───────────────────────────────────────────────────────

    def agg(self, by: list[str]) -> BookQuery:
        """
        Specify which dimensions to **collapse** and return a :class:`BookQuery`.

        ``by`` values: ``"MARKET"``, ``"CURRENCY"`` (case-insensitive).

        ``[]``                    → nothing collapsed → key ``(instr_id, market, currency)``
        ``["MARKET"]``            → collapse market  → key ``(instr_id, currency)``
        ``["CURRENCY"]``          → collapse currency → key ``(instr_id, market)``
        ``["MARKET","CURRENCY"]`` → fully collapsed  → key ``(instr_id,)``
        """
        return BookQuery(self, [d.upper() for d in by])

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_bid_ask(self, instr_id: str) -> tuple[float | None, float | None]:
        """Best ``(bid, ask)`` across all subscriptions for one instrument."""
        bid = self._bid.get(instr_id) if self._bid is not None else None
        ask = self._ask.get(instr_id) if self._ask is not None else None
        return (
            float(bid) if bid is not None and not np.isnan(bid) else None,
            float(ask) if ask is not None and not np.isnan(ask) else None,
        )


# ── Backward-compat alias ─────────────────────────────────────────────────────
LiveBook = CompositeBook
