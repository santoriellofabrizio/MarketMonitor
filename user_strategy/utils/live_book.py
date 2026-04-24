"""
CompositeBook: vectorised live book aggregator with pluggable filters and aggregation rules.

Terminology
-----------
sub_id        : subscription key used with market_data (e.g. "IM:IE00B4L5Y983")
instrument_id : canonical security key — required at registration
market        : venue / segment — required at registration
currency      : ISO currency code — required at registration
"""
from __future__ import annotations

from abc import ABC, abstractmethod
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


# ── Aggregation rules ─────────────────────────────────────────────────────────

class AggregationRule(ABC):
    """
    Vectorised rule that reduces a filtered book DataFrame to one value per group.

    Receives the sanitised book (index=sub_id, any field columns) and a `by`
    Series (sub_id → group key tuple), returns a Series indexed by those tuples.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series: ...


class FieldAgg(AggregationRule):
    """
    Aggregate a single field with any pandas groupby function.

    Examples
    --------
    FieldAgg("BID", "max")
    FieldAgg("LAST_PRICE", "last", name="last_price")
    FieldAgg("BID", np.median, name="median_bid")
    """

    def __init__(self, field: str, func: str | Callable, name: str | None = None):
        self._field = field
        self._func = func
        self._name = name or f"{func.__name__ if callable(func) else func}({field})"

    @property
    def name(self) -> str:
        return self._name

    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        return book[self._field].groupby(by).agg(self._func)


class BidAskMidAgg(AggregationRule):
    """
    Mid from independent bid-side and ask-side aggregations.

        result = (agg_bid(bid_field) + agg_ask(ask_field)) / 2

    Examples
    --------
    BidAskMidAgg("max", "min")    → best composite book
    BidAskMidAgg("mean", "mean")  → mean of average bid and average ask
    """

    def __init__(
        self,
        bid_agg: str | Callable = "max",
        ask_agg: str | Callable = "min",
        bid_field: str = "BID",
        ask_field: str = "ASK",
        name: str | None = None,
    ):
        self._bid_agg, self._ask_agg = bid_agg, ask_agg
        self._bid_field, self._ask_field = bid_field, ask_field
        ba = bid_agg.__name__ if callable(bid_agg) else bid_agg
        aa = ask_agg.__name__ if callable(ask_agg) else ask_agg
        self._name = name or f"mid({ba}({bid_field}),{aa}({ask_field}))"

    @property
    def name(self) -> str:
        return self._name

    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        bid = book[self._bid_field].groupby(by).agg(self._bid_agg)
        ask = book[self._ask_field].groupby(by).agg(self._ask_agg)
        return (bid + ask) / 2


class MeanMidsAgg(AggregationRule):
    """Mean of per-subscription (bid + ask) / 2, then averaged per group."""

    def __init__(self, bid_field: str = "BID", ask_field: str = "ASK",
                 name: str = "mean_mids"):
        self._bid_field, self._ask_field = bid_field, ask_field
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        return ((book[self._bid_field] + book[self._ask_field]) / 2).groupby(by).mean()


# ── Ready-made instances ──────────────────────────────────────────────────────

MAX_BID_MIN_ASK = BidAskMidAgg("max", "min", name="best_bid_ask")
MEAN_BID_ASK    = BidAskMidAgg("mean", "mean", name="mean_bid_ask")
MEAN_MIDS       = MeanMidsAgg(name="mean_mids")

_DEFAULT_RULES: list[AggregationRule] = [MAX_BID_MIN_ASK, MEAN_BID_ASK, MEAN_MIDS]


# ── BookResult ────────────────────────────────────────────────────────────────

class BookResult:
    """
    Immutable result of CompositeBook.agg(by=...).get_data(...).

    Carries aggregated prices and the `by` dimensions used, so it can
    produce the right output shape and a BookSnapshot compatible with
    BookStorage / TradeManager.

    Keys are always tuples:
        by=[]              → ("IE00B",)
        by=["CURRENCY"]    → ("IE00B", "EUR")
        by=["MARKET"]      → ("IE00B", "IM")
        by=["MARKET","CURRENCY"] → ("IE00B", "IM", "EUR")
    """

    def __init__(self, data: pd.Series, by: list[str]) -> None:
        # data: Series whose index values are tuples
        self._data = data
        self._by = by  # e.g. ["CURRENCY"] or ["MARKET", "CURRENCY"]

    # ── Conversion ────────────────────────────────────────────────────────────

    def as_dict(self) -> dict[tuple, float]:
        """Raw dict with tuple keys."""
        return self._data.to_dict()

    def as_series(self) -> pd.Series:
        """
        Flat Series (instr_id index) when by=[];
        MultiIndex Series (instr_id, dim…) otherwise.
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
        Single-column DataFrame when by=[];
        pivoted DataFrame (instr_id rows × last dim columns) otherwise.
        """
        s = self.as_series()
        if not self._by:
            return s.to_frame(name="mid")
        return s.unstack(level=-1)

    def as_snapshot(self) -> BookSnapshot:
        """
        Convert to BookSnapshot (dict[instr_id, FairvaluePrice]) for
        BookStorage.append() and TradeManager.

        Mapping logic:
          by=[]           → FairvaluePrice.scalar   per instr_id
          by=["CURRENCY"] → FairvaluePrice.by_currency per instr_id
          by=["MARKET"]   → FairvaluePrice.by_market   per instr_id
          other dims      → by_currency if CURRENCY present, else by_market
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
            # Multi-dim: prefer by_currency for TradeManager compat
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
    Pending query produced by CompositeBook.agg(by=[...]).

    Call .get_data() to execute and obtain a BookResult.
    """

    def __init__(self, book: "CompositeBook", by: list[str]) -> None:
        unknown = set(by) - _DIM_TO_COL.keys()
        if unknown:
            raise ValueError(
                f"Unknown dimension(s) {unknown}. Valid: {list(_DIM_TO_COL)}"
            )
        self._book = book
        self._by = by  # already upper-cased by CompositeBook.agg()

    def get_data(
        self,
        securities: list[str] | None = None,
        fx_rate: dict[str, float] | pd.Series | None = None,
        method: str | None = None,
    ) -> BookResult:
        """
        Execute the query and return a BookResult.

        Parameters
        ----------
        securities : optional list of instrument_ids to restrict the output.
        fx_rate    : optional currency → EUR rate mapping (dict or pd.Series).
                     Applied after filters, before aggregation.
                     e.g. {"EUR": 1.0, "USD": 0.92, "GBP": 1.16}
        method     : name of the AggregationRule to use (default: book.default_method).
        """
        book = self._book._clean_book
        if book is None or book.empty:
            return BookResult(pd.Series(dtype=float), self._by)

        if securities is not None:
            book = book[book[_COL_INSTR].isin(securities)]

        if book.empty:
            return BookResult(pd.Series(dtype=float), self._by)

        # FX conversion to EUR (applied after filters)
        if fx_rate is not None:
            bid_ask = [c for c in ("BID", "ASK") if c in book.columns]
            if bid_ask:
                fx = fx_rate if isinstance(fx_rate, dict) else fx_rate.to_dict()
                rates = book[_COL_CCY].map(fx).fillna(1.0)
                book = book.copy()
                book[bid_ask] = book[bid_ask].multiply(rates, axis=0)

        # Build groupby key: tuple of (instr_id, [market], [currency])
        key_cols = [_COL_INSTR] + [_DIM_TO_COL[d] for d in self._by]
        by_series = book[key_cols].apply(tuple, axis=1)

        # Select and run aggregation rule
        rule_name = method or self._book._default_method
        rule = next((r for r in self._book._agg_rules if r.name == rule_name), None)
        if rule is None:
            available = [r.name for r in self._book._agg_rules]
            raise KeyError(f"No rule {rule_name!r}. Available: {available}")

        try:
            result = rule.aggregate(book, by_series)
        except KeyError:
            result = pd.Series(dtype=float)

        return BookResult(result, self._by)


# ── CompositeBook ─────────────────────────────────────────────────────────────

class CompositeBook:
    """
    Aggregates live market data from one or more subscriptions per instrument.

    Parameters
    ----------
    default_method : AggregationRule name used when method= is omitted in get_data().
    rules          : aggregation rules applied in every update(). Defaults to
                     [MAX_BID_MIN_ASK, MEAN_BID_ASK, MEAN_MIDS].

    ---

    USAGE GUIDE
    ===========

    1. Registration — instr_id, market and currency are all mandatory
    ----------------------------------------------------------------

        book = CompositeBook()

        # single subscription
        book.register("IM:IE00B4L5Y983",
                      instr_id="IE00B4L5Y983", market="IM", currency="EUR")

        # same instrument across multiple markets / currencies
        for mkt, ccy in [("IM", "EUR"), ("FP", "EUR"), ("NA", "USD")]:
            book.register(f"{mkt}:IE00B4L5Y983",
                          instr_id="IE00B4L5Y983", market=mkt, currency=ccy)

        # bulk from instrument objects (.id, .market, .currency must exist)
        book.register_from_instruments({inst.id: inst for inst in my_instruments})

        # chainable
        book = (
            CompositeBook()
            .register("ITRAXX.S42.5Y", instr_id="ITRAXX.S42.5Y",
                      market="OTC", currency="EUR")
            .register("CDXIG.43.5Y",   instr_id="CDXIG.43.5Y",
                      market="OTC", currency="USD")
        )

    2. Filters
    ----------

        from market_monitor.utils.book_utils import SpreadEWMA, PriceEWMA

        book.add_filter(SpreadEWMA(tau_seconds=600, max_multiplier=2.0))  # global
        book.add_filter(PriceEWMA(tau_seconds=300, max_ret=0.005),
                        securities=["IM:IE00B4L5Y983"])                   # scoped

    3. Aggregation rules
    --------------------

        book.add_aggregation(FieldAgg("LAST_PRICE", "last", name="last_price"))
        book.add_aggregation(BidAskMidAgg("mean", "mean", name="vwap_mid"))

        book.available_methods  # → ["best_bid_ask", "mean_bid_ask", "mean_mids", ...]

    4. Hot path — update
    ----------------------

        raw = market_data.get_data_field(field=["BID", "ASK"])
        book.update(raw)

    5. Querying — agg / get_data / BookResult
    ------------------------------------------

        # aggregate over all markets and currencies → one mid per ISIN
        result = book.agg(by=[]).get_data()

        # keep currency as separate dimension → one mid per (ISIN, currency)
        result = book.agg(by=["CURRENCY"]).get_data()

        # keep both → one mid per (ISIN, market, currency)
        result = book.agg(by=["MARKET", "CURRENCY"]).get_data()

        # filter to a subset and convert non-EUR to EUR
        fx = {"EUR": 1.0, "USD": 0.92, "GBP": 1.16}
        result = (
            book.agg(by=[])
                .get_data(securities=my_isin_list, fx_rate=fx, method="best_bid_ask")
        )

    6. Consuming BookResult
    -----------------------

        result.as_series()
        # by=[]           → pd.Series indexed by instr_id
        # by=["CURRENCY"] → pd.Series with MultiIndex (instr_id, currency)

        result.as_df()
        # by=[]           → single-column DataFrame
        # by=["CURRENCY"] → DataFrame with instr_id rows × currency columns

        result.as_dict()
        # → {("IE00B", "EUR"): 99.5, ("IE00B", "USD"): 98.2, ...}

        result.as_snapshot()
        # → BookSnapshot compatible with BookStorage.append() and TradeManager
        # by=[]           → {instr_id: FairvaluePrice.scalar(...)}
        # by=["CURRENCY"] → {instr_id: FairvaluePrice.by_currency(...)}
        # by=["MARKET"]   → {instr_id: FairvaluePrice.by_market(...)}

        # typical usage with BookStorage
        book_storage.append(result.as_snapshot())

    7. Per-instrument bid / ask
    ---------------------------

        bid, ask = book.get_bid_ask("IE00B4L5Y983")
        # → (float | None, float | None) — best bid/ask across all subscriptions
    """

    def __init__(
        self,
        default_method: str = "best_bid_ask",
        rules: list[AggregationRule] | None = None,
    ) -> None:
        self._default_method = default_method
        self._agg_rules: list[AggregationRule] = (
            list(rules) if rules is not None else list(_DEFAULT_RULES)
        )

        self._sub_to_instr: dict[str, str] = {}
        self._sub_metadata: dict[str, tuple[str, str]] = {}  # sub_id → (market, currency)

        self._filters: list[tuple[BookFilter, frozenset[str] | None]] = []

        # Set by update(); consumed by BookQuery.get_data()
        self._clean_book: pd.DataFrame | None = None

        # Best bid/ask per instrument_id for get_bid_ask()
        self._bid: pd.Series | None = None
        self._ask: pd.Series | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        sub_id: str,
        instr_id: str,
        market: str,
        currency: str,
    ) -> "CompositeBook":
        """Register one subscription. All four arguments are required."""
        self._sub_to_instr[sub_id] = instr_id
        self._sub_metadata[sub_id] = (market, currency)
        return self

    def register_from_instruments(self, instruments: dict) -> "CompositeBook":
        """
        Bulk-register from a {sub_id: instr_obj} dict.

        instr_obj must expose .id, .market and .currency attributes.
        """
        for sub_id, instr in instruments.items():
            self.register(
                sub_id=sub_id,
                instr_id=instr.id,
                market=instr.market,
                currency=instr.currency,
            )
        return self

    # ── Configuration ─────────────────────────────────────────────────────────

    def add_filter(
        self,
        filt: BookFilter,
        securities: Sequence[str] | None = None,
    ) -> "CompositeBook":
        """Attach a BookFilter. securities=None → global; otherwise scoped to those sub_ids."""
        self._filters.append(
            (filt, frozenset(securities) if securities is not None else None)
        )
        return self

    def add_aggregation(self, rule: AggregationRule) -> "CompositeBook":
        """Append a custom AggregationRule, computed on every get_data() call."""
        self._agg_rules.append(rule)
        return self

    # ── Update (hot path) ─────────────────────────────────────────────────────

    def update(self, raw: pd.DataFrame) -> None:
        """
        Ingest a market data snapshot and apply filters.

        Parameters
        ----------
        raw : DataFrame with index=sub_id, any field columns (BID, ASK, …).
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

        # --- Apply filter chain (BID + ASK required per filter) --------------
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

        # --- Restrict to registered sub_ids and attach metadata --------------
        known = book.index.intersection(self._sub_to_instr.keys())
        book = book.reindex(known).copy()
        book[_COL_INSTR] = pd.Series(self._sub_to_instr).reindex(book.index)
        book[_COL_MKT]   = pd.Series({s: self._sub_metadata[s][0] for s in book.index})
        book[_COL_CCY]   = pd.Series({s: self._sub_metadata[s][1] for s in book.index})
        self._clean_book = book.dropna(subset=[_COL_INSTR])

        # --- Best bid / ask per instr_id (for get_bid_ask) -------------------
        instr_by = self._clean_book[_COL_INSTR]
        if "BID" in self._clean_book.columns:
            self._bid = self._clean_book["BID"].groupby(instr_by).max()
        if "ASK" in self._clean_book.columns:
            self._ask = self._clean_book["ASK"].groupby(instr_by).min()

    # ── Query interface ───────────────────────────────────────────────────────

    def agg(self, by: list[str]) -> BookQuery:
        """
        Define groupby dimensions and return a BookQuery.

        Parameters
        ----------
        by : extra dimensions to keep distinct beyond instr_id.
             Valid values: "MARKET", "CURRENCY" (case-insensitive).
             []                 → one row per instr_id
             ["CURRENCY"]       → one row per (instr_id, currency)
             ["MARKET"]         → one row per (instr_id, market)
             ["MARKET","CURRENCY"] → one row per (instr_id, market, currency)
        """
        return BookQuery(self, [d.upper() for d in by])

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_bid_ask(self, instr_id: str) -> tuple[float | None, float | None]:
        """Return (best_bid, best_ask) for one instrument across all its subscriptions."""
        bid = self._bid.get(instr_id) if self._bid is not None else None
        ask = self._ask.get(instr_id) if self._ask is not None else None
        return (
            float(bid) if bid is not None and not np.isnan(bid) else None,
            float(ask) if ask is not None and not np.isnan(ask) else None,
        )

    @property
    def available_methods(self) -> list[str]:
        """Names of all registered aggregation rules."""
        return [r.name for r in self._agg_rules]


# ── Backward-compat alias ─────────────────────────────────────────────────────
LiveBook = CompositeBook
