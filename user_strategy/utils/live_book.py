"""
LiveBook: vectorised live book aggregator with pluggable filters and aggregation rules.

Terminology
-----------
sub_id        : subscription key used with market_data (e.g. "IM:IE00B4L5Y983")
instrument_id : canonical security key for groupby / output (required at registration)

When sub_id == instrument_id (1-to-1), the groupby is a pass-through.
When multiple sub_ids map to the same instrument_id (multi-market), the groupby aggregates.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from market_monitor.utils.book_utils import BookFilter


# ── Aggregation rules ─────────────────────────────────────────────────────────

class AggregationRule(ABC):
    """
    Vectorised rule that reduces a (filtered) book DataFrame to one value per instrument.

    Subclasses receive the full sanitised book and the sub_id → instrument_id
    groupby Series; they return a Series indexed by instrument_id.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key used in get_mid(method=...) lookups."""
        ...

    @abstractmethod
    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        """
        Parameters
        ----------
        book : sanitised DataFrame (zeros removed); index=sub_id, columns=any fields.
        by   : Series mapping sub_id → instrument_id.

        Returns pd.Series indexed by instrument_id.
        """
        ...


class FieldAgg(AggregationRule):
    """
    Aggregate a single field with any pandas groupby aggregation.

    Examples
    --------
    FieldAgg("BID", "max")          → max bid per instrument
    FieldAgg("LAST_PRICE", "last")  → last price per instrument
    FieldAgg("BID", np.median)      → median bid per instrument
    """

    def __init__(
        self,
        field: str,
        func: str | Callable,
        name: str | None = None,
    ):
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
    Mid price from independent bid-side and ask-side aggregations.

    result = (agg_bid(bid_field) + agg_ask(ask_field)) / 2

    Parameters
    ----------
    bid_agg, ask_agg : pandas groupby aggregation ("max", "min", "mean", …) or callable.
    bid_field        : column name for bids (default "BID").
    ask_field        : column name for asks (default "ASK").

    Examples
    --------
    BidAskMidAgg("max", "min")   → best composite book (tightest spread)
    BidAskMidAgg("mean", "mean") → mean of average bid and average ask
    """

    def __init__(
        self,
        bid_agg: str | Callable = "max",
        ask_agg: str | Callable = "min",
        bid_field: str = "BID",
        ask_field: str = "ASK",
        name: str | None = None,
    ):
        self._bid_agg = bid_agg
        self._ask_agg = ask_agg
        self._bid_field = bid_field
        self._ask_field = ask_field
        _ba = bid_agg.__name__ if callable(bid_agg) else bid_agg
        _aa = ask_agg.__name__ if callable(ask_agg) else ask_agg
        self._name = name or f"mid({_ba}({bid_field}), {_aa}({ask_field}))"

    @property
    def name(self) -> str:
        return self._name

    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        bid = book[self._bid_field].groupby(by).agg(self._bid_agg)
        ask = book[self._ask_field].groupby(by).agg(self._ask_agg)
        return (bid + ask) / 2


class MeanMidsAgg(AggregationRule):
    """
    Mean of per-subscription (bid + ask) / 2, then averaged per instrument.

    Differs from BidAskMidAgg("mean","mean") when there are multiple
    subscriptions per instrument: here the averaging happens on the
    per-subscription mid, not on the pooled bid/ask sides separately.

    Parameters
    ----------
    bid_field : column for bids (default "BID").
    ask_field : column for asks (default "ASK").
    """

    def __init__(
        self,
        bid_field: str = "BID",
        ask_field: str = "ASK",
        name: str = "mean_mids",
    ):
        self._bid_field = bid_field
        self._ask_field = ask_field
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        per_sub = (book[self._bid_field] + book[self._ask_field]) / 2
        return per_sub.groupby(by).mean()


# ── Ready-made rule instances ─────────────────────────────────────────────────

MAX_BID_MIN_ASK = BidAskMidAgg(bid_agg="max", ask_agg="min", name="best_bid_ask")
MEAN_BID_ASK    = BidAskMidAgg(bid_agg="mean", ask_agg="mean", name="mean_bid_ask")
MEAN_MIDS       = MeanMidsAgg(name="mean_mids")

_DEFAULT_RULES: list[AggregationRule] = [MAX_BID_MIN_ASK, MEAN_BID_ASK, MEAN_MIDS]


# ── LiveBook ──────────────────────────────────────────────────────────────────

class LiveBook:
    """
    Aggregates live market data from one or more subscriptions per instrument.

    Parameters
    ----------
    default_method : name of the AggregationRule used by get_mid() with no argument.
    rules          : aggregation rules applied on every update().
                     Defaults to [MAX_BID_MIN_ASK, MEAN_BID_ASK, MEAN_MIDS].

    ---

    USAGE GUIDE
    ===========

    1. Registration
    ---------------
    Every subscription (sub_id) must be mapped to a mandatory instrument_id.
    market and currency are optional metadata stored for reference.

        lb = LiveBook()

        # one subscription → one instrument (1-to-1)
        lb.register("IE00B4L5Y983", instrument_id="IE00B4L5Y983", market="IM", currency="EUR")

        # multiple markets → same instrument (many-to-one aggregation)
        for mkt in ["IM", "FP", "NA"]:
            lb.register(f"{mkt}:IE00B4L5Y983", instrument_id="IE00B4L5Y983",
                        market=mkt, currency="EUR")

        # bulk registration from instrument objects (need .id, optionally .market/.currency)
        lb.register_from_instruments({inst.id: inst for inst in my_instruments})

        # all methods return self → chainable
        lb = (
            LiveBook()
            .register("ITRAXX.S42.5Y", instrument_id="ITRAXX.S42.5Y")
            .register("CDXIG.43.5Y",   instrument_id="CDXIG.43.5Y")
        )

    2. Filters
    ----------
    Filters conform to the BookFilter protocol (update + get_valid_book).
    They are applied in registration order before aggregation.

        from market_monitor.utils.book_utils import SpreadEWMA, PriceEWMA

        # global filter — applied to every sub_id
        lb.add_filter(SpreadEWMA(tau_seconds=600, max_multiplier=2.0))

        # scoped filter — only for a subset of sub_ids
        lb.add_filter(PriceEWMA(tau_seconds=300, max_ret=0.005),
                      securities=["IM:IE00B4L5Y983", "FP:IE00B4L5Y983"])

        # filters are chained: second filter sees output of first

    3. Aggregation rules
    --------------------
    Three rules are registered by default (accessible via module-level presets):

        MAX_BID_MIN_ASK  →  (max(BID) + min(ASK)) / 2  per instrument  ["best_bid_ask"]
        MEAN_BID_ASK     →  (mean(BID) + mean(ASK)) / 2                ["mean_bid_ask"]
        MEAN_MIDS        →  mean((BID + ASK) / 2)                      ["mean_mids"]

    Add custom rules at construction or at runtime:

        from user_strategy.utils.live_book import FieldAgg, BidAskMidAgg, MeanMidsAgg

        # override default rules entirely
        lb = LiveBook(rules=[MAX_BID_MIN_ASK])

        # add a rule for any field/function
        lb.add_aggregation(FieldAgg("LAST_PRICE", "last", name="last_price"))
        lb.add_aggregation(FieldAgg("BID", "median", name="median_bid"))

        # custom BidAsk mid with non-standard columns
        lb.add_aggregation(BidAskMidAgg("mean", "mean",
                                        bid_field="BID", ask_field="ASK",
                                        name="my_mean_mid"))

        # mean of per-subscription mids on non-standard columns
        lb.add_aggregation(MeanMidsAgg(bid_field="BID", ask_field="ASK",
                                       name="my_mean_mids"))

        # check registered rule names
        lb.available_methods  # → ["best_bid_ask", "mean_bid_ask", "mean_mids", "last_price", ...]

    4. Hot path — update + get_mid
    -------------------------------
    Call update() every tick with a DataFrame indexed by sub_id.
    Zeros are treated as NaN. Columns can be any fields (BID, ASK, LAST_PRICE, …).

        raw = market_data.get_data_field(field=["BID", "ASK"])
        lb.update(raw)

        mid = lb.get_mid()                    # uses default_method ("best_bid_ask")
        mid = lb.get_mid("mean_mids")         # explicit method name
        mid = lb.get_mid("last_price")        # custom rule added earlier

        # mid is a pd.Series indexed by instrument_id
        self.book_mid.update(mid)

        # LAST_PRICE fallback outside LiveBook
        last = market_data.get_data_field(field="LAST_PRICE")
        mid  = lb.get_mid().combine_first(last)

    5. Per-instrument bid/ask
    -------------------------
        bid, ask = lb.get_bid_ask("IE00B4L5Y983")
        # returns (float | None, float | None) — None if no valid quote
    """

    def __init__(
        self,
        default_method: str = "best_bid_ask",
        rules: list[AggregationRule] | None = None,
    ) -> None:
        self._default_method = default_method
        self._agg_rules: list[AggregationRule] = list(rules) if rules is not None else list(_DEFAULT_RULES)

        self._sub_to_instr: dict[str, str] = {}
        self._sub_metadata: dict[str, tuple[str | None, str | None]] = {}  # sub_id → (market, currency)
        self._instr_series: pd.Series | None = None  # built lazily; index=sub_id, values=instrument_id

        self._filters: list[tuple[BookFilter, frozenset[str] | None]] = []

        self._bid: pd.Series | None = None   # per-instrument best bid (for get_bid_ask)
        self._ask: pd.Series | None = None   # per-instrument best ask
        self._mid_cache: dict[str, pd.Series] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        sub_id: str,
        instrument_id: str,
        market: str | None = None,
        currency: str | None = None,
    ) -> "LiveBook":
        """
        Register one subscription.

        Parameters
        ----------
        sub_id        : key used by market_data (e.g. "IM:IE00B4L5Y983").
        instrument_id : canonical security identifier (required).
        market        : optional venue / market metadata.
        currency      : optional currency metadata.
        """
        self._sub_to_instr[sub_id] = instrument_id
        self._sub_metadata[sub_id] = (market, currency)
        self._instr_series = None
        return self

    def register_from_instruments(self, instruments: dict) -> "LiveBook":
        """
        Bulk-register from a {sub_id: instr_obj} dict.

        Reads instr_obj.id (required), instr_obj.market and instr_obj.currency (optional).
        """
        for sub_id, instr in instruments.items():
            self.register(
                sub_id=sub_id,
                instrument_id=instr.id,
                market=getattr(instr, "market", None),
                currency=getattr(instr, "currency", None),
            )
        return self

    # ── Configuration ─────────────────────────────────────────────────────────

    def add_filter(
        self,
        filt: BookFilter,
        securities: Sequence[str] | None = None,
    ) -> "LiveBook":
        """
        Attach a BookFilter.

        Parameters
        ----------
        filt       : BookFilter (SpreadEWMA, PriceEWMA, …).
        securities : Sub IDs this filter applies to. None → global (all rows).
        """
        self._filters.append((filt, frozenset(securities) if securities is not None else None))
        return self

    def add_aggregation(self, rule: AggregationRule) -> "LiveBook":
        """Append a custom AggregationRule, computed on every update()."""
        self._agg_rules.append(rule)
        return self

    # ── Update (hot path) ─────────────────────────────────────────────────────

    def update(self, raw: pd.DataFrame) -> None:
        """
        Ingest a market data snapshot, apply filters, run all aggregation rules.

        Parameters
        ----------
        raw : DataFrame with index=sub_id, any field columns (BID, ASK, LAST_PRICE, …).
              Zero values are treated as NaN.
        """
        if self._instr_series is None:
            self._instr_series = pd.Series(self._sub_to_instr, dtype="object")

        self._mid_cache.clear()

        if raw.empty:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        # --- Sanitise: replace 0 with NaN, drop rows that are entirely NaN --
        book = raw.replace({0: np.nan}).dropna(how="all")

        if book.empty:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        # --- Apply filter chain (BID+ASK required per filter) ---------------
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
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        # --- Map sub_id → instrument_id (drop unknown subs) -----------------
        known = self._instr_series.reindex(book.index).dropna()
        book = book.reindex(known.index)
        instr = known  # Series: sub_id → instrument_id

        # --- Best bid / best ask (for get_bid_ask) ---------------------------
        self._bid = book["BID"].groupby(instr).max() if "BID" in book.columns else pd.Series(dtype=float)
        self._ask = book["ASK"].groupby(instr).min() if "ASK" in book.columns else pd.Series(dtype=float)

        # --- Run all aggregation rules ---------------------------------------
        for rule in self._agg_rules:
            try:
                self._mid_cache[rule.name] = rule.aggregate(book, instr)
            except KeyError:
                pass  # required field not present in this snapshot

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_mid(self, method: str | None = None) -> pd.Series:
        """
        Return per-instrument aggregated prices.

        Parameters
        ----------
        method : name of an AggregationRule (or None → default_method).
        """
        m = method if method is not None else self._default_method
        result = self._mid_cache.get(m)
        if result is None:
            available = list(self._mid_cache)
            raise KeyError(
                f"No result for aggregation {m!r}. "
                f"Available: {available} — did you call update() first?"
            )
        return result

    def get_bid_ask(self, instrument_id: str) -> tuple[float | None, float | None]:
        """Return (best_bid, best_ask) for a single instrument."""
        bid = self._bid.get(instrument_id) if self._bid is not None else None
        ask = self._ask.get(instrument_id) if self._ask is not None else None
        return (
            float(bid) if bid is not None and not np.isnan(bid) else None,
            float(ask) if ask is not None and not np.isnan(ask) else None,
        )

    @property
    def available_methods(self) -> list[str]:
        """Names of all registered aggregation rules."""
        return [r.name for r in self._agg_rules]
