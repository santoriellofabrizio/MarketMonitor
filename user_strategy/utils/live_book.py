"""
LiveBook: vectorised live book aggregator with pluggable filters and aggregation rules.

Terminology
-----------
sub_id        : subscription key used with market_data (e.g. "IM:IE00B4L5Y983")
instrument_id : canonical security key — required at registration
market        : venue / segment — required at registration
currency      : ISO currency code — required at registration
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from market_monitor.utils.book_utils import BookFilter

_AggregateBy = Literal["isin", "isin_market", "isin_currency"]


# ── Aggregation rules ─────────────────────────────────────────────────────────

class AggregationRule(ABC):
    """
    Vectorised rule that reduces a (filtered) book DataFrame to one value per group.

    Subclasses receive the sanitised book DataFrame and the pre-built `by` Series
    (sub_id → group key, where the group key depends on the LiveBook aggregate_by
    setting: a plain instrument_id string, or "isin:market", or "isin:currency").
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
        book : sanitised DataFrame (zeros and filter-invalid rows removed);
               index=sub_id, columns=any market data fields.
        by   : Series mapping sub_id → group key (instrument_id or composite).

        Returns pd.Series indexed by the group key.
        """
        ...


class FieldAgg(AggregationRule):
    """
    Aggregate a single field with any pandas groupby aggregation.

    Examples
    --------
    FieldAgg("BID", "max")           → max bid per group
    FieldAgg("LAST_PRICE", "last")   → last price per group
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
    Mid price from independent bid-side and ask-side aggregations.

        result = (agg_bid(bid_field) + agg_ask(ask_field)) / 2

    Parameters
    ----------
    bid_agg, ask_agg : pandas groupby aggregation ("max", "min", "mean", …) or callable.
    bid_field        : column name for bids (default "BID").
    ask_field        : column name for asks (default "ASK").

    Examples
    --------
    BidAskMidAgg("max", "min")    → best composite book (tightest spread)
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
        self._bid_agg = bid_agg
        self._ask_agg = ask_agg
        self._bid_field = bid_field
        self._ask_field = ask_field
        _ba = bid_agg.__name__ if callable(bid_agg) else bid_agg
        _aa = ask_agg.__name__ if callable(ask_agg) else ask_agg
        self._name = name or f"mid({_ba}({bid_field}),{_aa}({ask_field}))"

    @property
    def name(self) -> str:
        return self._name

    def aggregate(self, book: pd.DataFrame, by: pd.Series) -> pd.Series:
        bid = book[self._bid_field].groupby(by).agg(self._bid_agg)
        ask = book[self._ask_field].groupby(by).agg(self._ask_agg)
        return (bid + ask) / 2


class MeanMidsAgg(AggregationRule):
    """
    Mean of per-subscription (bid + ask) / 2, then averaged per group.

    Differs from BidAskMidAgg("mean","mean") when multiple subscriptions exist:
    here the averaging is on per-subscription mids, not on pooled bid/ask sides.
    """

    def __init__(self, bid_field: str = "BID", ask_field: str = "ASK",
                 name: str = "mean_mids"):
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
    default_method : name of the AggregationRule returned by get_mid() with no argument.
    rules          : aggregation rules applied on every update().
                     Defaults to [MAX_BID_MIN_ASK, MEAN_BID_ASK, MEAN_MIDS].
    aggregate_by   : groupby key used when building per-instrument results.
                     "isin"          → one row per instrument_id (default)
                     "isin_market"   → one row per (instrument_id, market) pair
                     "isin_currency" → one row per (instrument_id, currency) pair

    ---

    USAGE GUIDE
    ===========

    1. Registration — instrument_id, market and currency are all mandatory
    -------------------------------------------------------------------

        lb = LiveBook()

        # single subscription
        lb.register("IM:IE00B4L5Y983",
                    instrument_id="IE00B4L5Y983", market="IM", currency="EUR")

        # same instrument across multiple markets
        for mkt, ccy in [("IM", "EUR"), ("FP", "EUR"), ("NA", "USD")]:
            lb.register(f"{mkt}:IE00B4L5Y983",
                        instrument_id="IE00B4L5Y983", market=mkt, currency=ccy)

        # bulk from instrument objects (.id, .market, .currency must exist)
        lb.register_from_instruments({inst.id: inst for inst in my_instruments})

        # all methods return self → chainable
        lb = (
            LiveBook()
            .register("ITRAXX.S42.5Y", instrument_id="ITRAXX.S42.5Y",
                      market="OTC", currency="EUR")
            .register("CDXIG.43.5Y",   instrument_id="CDXIG.43.5Y",
                      market="OTC", currency="USD")
        )

    2. Groupby mode
    ---------------
    Controls how subscriptions are collapsed into output rows.

        # one mid per ISIN (default) — aggregates across all markets and currencies
        lb = LiveBook(aggregate_by="isin")

        # one mid per (ISIN, market) — separate rows for IM, FP, NA
        lb = LiveBook(aggregate_by="isin_market")
        mid = lb.get_mid()
        # mid.index → ["IE00B4L5Y983:IM", "IE00B4L5Y983:FP", ...]

        # one mid per (ISIN, currency) — separate rows for EUR, USD
        lb = LiveBook(aggregate_by="isin_currency")
        mid = lb.get_mid()
        # mid.index → ["IE00B4L5Y983:EUR", "IE00B4L5Y983:USD", ...]

    3. FX conversion — normalise to EUR before aggregating
    -------------------------------------------------------
    When aggregate_by="isin" and subscriptions have different currencies,
    pass fx_rate to update() to convert all prices to EUR before the groupby.
    fx_rate is a dict or pd.Series mapping currency → EUR rate.

        fx = {"EUR": 1.0, "USD": 0.92, "GBP": 1.16}

        lb.update(raw, fx_rate=fx)   # prices converted to EUR before aggregation
        mid = lb.get_mid()           # result is in EUR

    FX conversion is applied after filters (filters see original-currency prices)
    and before aggregation rules (rules always operate on converted prices when
    fx_rate is provided).

    4. Filters
    ----------
    Conform to BookFilter protocol (update + get_valid_book).
    Applied in registration order before aggregation.

        from market_monitor.utils.book_utils import SpreadEWMA, PriceEWMA

        lb.add_filter(SpreadEWMA(tau_seconds=600, max_multiplier=2.0))   # global
        lb.add_filter(PriceEWMA(tau_seconds=300, max_ret=0.005),
                      securities=["IM:IE00B4L5Y983"])                    # scoped

    5. Aggregation rules
    --------------------
    Three rules are active by default:

        MAX_BID_MIN_ASK  →  (max(BID) + min(ASK)) / 2   ["best_bid_ask"]
        MEAN_BID_ASK     →  (mean(BID) + mean(ASK)) / 2 ["mean_bid_ask"]
        MEAN_MIDS        →  mean((BID+ASK)/2)            ["mean_mids"]

    Add custom rules:

        from user_strategy.utils.live_book import FieldAgg, BidAskMidAgg

        lb.add_aggregation(FieldAgg("LAST_PRICE", "last", name="last_price"))
        lb.add_aggregation(FieldAgg("BID", "median", name="median_bid"))
        lb.add_aggregation(BidAskMidAgg("mean", "mean", name="vwap_mid"))

        lb.available_methods  # → ["best_bid_ask", "mean_bid_ask", "mean_mids", ...]

        # override default rules entirely
        lb = LiveBook(rules=[MAX_BID_MIN_ASK])

    6. Hot path — update + get_mid
    --------------------------------

        raw = market_data.get_data_field(field=["BID", "ASK"])
        lb.update(raw)                      # no FX conversion
        lb.update(raw, fx_rate=fx_dict)     # with FX conversion to EUR

        mid = lb.get_mid()                  # default method
        mid = lb.get_mid("mean_mids")       # explicit method
        mid = lb.get_mid("last_price")      # custom rule

        # LAST_PRICE fallback (outside LiveBook)
        last = market_data.get_data_field(field="LAST_PRICE")
        mid  = lb.get_mid().combine_first(last)

    7. Per-instrument bid/ask
    -------------------------

        bid, ask = lb.get_bid_ask("IE00B4L5Y983")
        # → (float | None, float | None); best bid and best ask across all subs
    """

    def __init__(
        self,
        default_method: str = "best_bid_ask",
        rules: list[AggregationRule] | None = None,
        aggregate_by: _AggregateBy = "isin",
    ) -> None:
        self._default_method = default_method
        self._agg_rules: list[AggregationRule] = list(rules) if rules is not None else list(_DEFAULT_RULES)
        self._aggregate_by: _AggregateBy = aggregate_by

        self._sub_to_instr: dict[str, str] = {}
        self._sub_metadata: dict[str, tuple[str, str]] = {}  # sub_id → (market, currency)

        # lazily built; invalidated on every register() call
        self._instr_series: pd.Series | None = None  # sub_id → instrument_id
        self._by_series: pd.Series | None = None     # sub_id → groupby key

        self._filters: list[tuple[BookFilter, frozenset[str] | None]] = []

        self._bid: pd.Series | None = None   # best bid per instrument (for get_bid_ask)
        self._ask: pd.Series | None = None   # best ask per instrument
        self._mid_cache: dict[str, pd.Series] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        sub_id: str,
        instrument_id: str,
        market: str,
        currency: str,
    ) -> "LiveBook":
        """Register one subscription. All four arguments are required."""
        self._sub_to_instr[sub_id] = instrument_id
        self._sub_metadata[sub_id] = (market, currency)
        self._instr_series = None
        self._by_series = None
        return self

    def register_from_instruments(self, instruments: dict) -> "LiveBook":
        """
        Bulk-register from a {sub_id: instr_obj} dict.

        instr_obj must have .id, .market and .currency attributes.
        """
        for sub_id, instr in instruments.items():
            self.register(
                sub_id=sub_id,
                instrument_id=instr.id,
                market=instr.market,
                currency=instr.currency,
            )
        return self

    # ── Configuration ─────────────────────────────────────────────────────────

    def add_filter(
        self,
        filt: BookFilter,
        securities: Sequence[str] | None = None,
    ) -> "LiveBook":
        """Attach a BookFilter. securities=None → global; otherwise scoped to those sub_ids."""
        self._filters.append((filt, frozenset(securities) if securities is not None else None))
        return self

    def add_aggregation(self, rule: AggregationRule) -> "LiveBook":
        """Append a custom AggregationRule, computed on every update()."""
        self._agg_rules.append(rule)
        return self

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_series(self) -> None:
        self._instr_series = pd.Series(self._sub_to_instr, dtype="object")
        if self._aggregate_by == "isin":
            self._by_series = self._instr_series
        elif self._aggregate_by == "isin_market":
            self._by_series = pd.Series(
                {s: f"{i}:{self._sub_metadata[s][0]}" for s, i in self._sub_to_instr.items()},
                dtype="object",
            )
        else:  # isin_currency
            self._by_series = pd.Series(
                {s: f"{i}:{self._sub_metadata[s][1]}" for s, i in self._sub_to_instr.items()},
                dtype="object",
            )

    # ── Update (hot path) ─────────────────────────────────────────────────────

    def update(
        self,
        raw: pd.DataFrame,
        fx_rate: dict[str, float] | pd.Series | None = None,
    ) -> None:
        """
        Ingest a market data snapshot, apply filters, run all aggregation rules.

        Parameters
        ----------
        raw     : DataFrame with index=sub_id, any field columns (BID, ASK, LAST_PRICE, …).
                  Zero values are treated as NaN.
        fx_rate : optional currency → EUR rate mapping applied after filters and before
                  aggregation (converts BID/ASK to EUR). Useful when aggregate_by="isin"
                  and subscriptions trade in different currencies.
                  e.g. {"EUR": 1.0, "USD": 0.92, "GBP": 1.16}
        """
        if self._instr_series is None:
            self._build_series()

        self._mid_cache.clear()

        if raw.empty:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        book = raw.replace({0: np.nan}).dropna(how="all")

        if book.empty:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        # --- Apply filter chain (requires BID + ASK) -------------------------
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

        # --- FX conversion to EUR (applied after filters) --------------------
        if fx_rate is not None:
            bid_ask = [c for c in ("BID", "ASK") if c in book.columns]
            if bid_ask:
                ccy_series = pd.Series(
                    {s: self._sub_metadata[s][1] for s in book.index if s in self._sub_metadata},
                    dtype="object",
                )
                rates = ccy_series.map(fx_rate if isinstance(fx_rate, dict) else fx_rate.to_dict()).fillna(1.0)
                book = book.copy()
                book[bid_ask] = book[bid_ask].multiply(rates, axis=0)

        # --- Map sub_ids to groupby keys -------------------------------------
        known_instr = self._instr_series.reindex(book.index).dropna()
        known_by    = self._by_series.reindex(book.index).dropna()
        book = book.reindex(known_by.index)

        # --- Best bid / ask per instrument (always by isin, for get_bid_ask) -
        self._bid = book["BID"].groupby(known_instr.reindex(known_by.index)).max() \
            if "BID" in book.columns else pd.Series(dtype=float)
        self._ask = book["ASK"].groupby(known_instr.reindex(known_by.index)).min() \
            if "ASK" in book.columns else pd.Series(dtype=float)

        # --- Run all aggregation rules ---------------------------------------
        for rule in self._agg_rules:
            try:
                self._mid_cache[rule.name] = rule.aggregate(book, known_by)
            except KeyError:
                pass  # required field absent in this snapshot

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_mid(self, method: str | None = None) -> pd.Series:
        """
        Return aggregated prices per group (instrument_id, or composite key).

        Parameters
        ----------
        method : AggregationRule name, or None to use default_method.
        """
        m = method if method is not None else self._default_method
        result = self._mid_cache.get(m)
        if result is None:
            raise KeyError(
                f"No result for {m!r}. "
                f"Available: {list(self._mid_cache)} — did you call update() first?"
            )
        return result

    def get_bid_ask(self, instrument_id: str) -> tuple[float | None, float | None]:
        """Return (best_bid, best_ask) for a single instrument across all its subscriptions."""
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
