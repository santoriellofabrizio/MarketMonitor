"""
LiveBook: vectorised live BID/ASK aggregator with pluggable filters.

Terminology
-----------
sub_id       : subscription key used with market_data (e.g. "IE00B4L5Y983")
instrument_id: canonical security key for groupby/output (defaults to sub_id)

When sub_id == instrument_id (CreditPriceEngine, 1-to-1), the groupby is a pass-through.
When multiple sub_ids map to the same instrument_id (multi-market), the groupby aggregates.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from market_monitor.utils.book_utils import BookFilter

_METHODS = frozenset({"best_bid_ask", "mean_bid_ask", "mean_mids"})


class LiveBook:
    """
    Aggregates live BID/ASK data from one or more subscriptions per instrument.

    Parameters
    ----------
    default_method : {"best_bid_ask", "mean_bid_ask", "mean_mids"}
        Aggregation method used by get_mid() when no method is supplied.
        - best_bid_ask : max(bids) and min(asks) across subscriptions
        - mean_bid_ask : mean(bids) and mean(asks) across subscriptions
        - mean_mids    : mean of per-subscription (BID+ASK)/2
    """

    def __init__(self, default_method: str = "best_bid_ask") -> None:
        if default_method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {default_method!r}")
        self._default_method = default_method

        self._sub_to_instr: dict[str, str] = {}
        self._sub_metadata: dict[str, tuple[str | None, str | None]] = {}  # sub_id → (market, currency)
        self._instr_series: pd.Series | None = None  # built lazily, index=sub_id, values=instrument_id

        # (filter_obj, frozenset_of_sub_ids | None); None scope = global
        self._filters: list[tuple[BookFilter, frozenset[str] | None]] = []

        self._bid: pd.Series | None = None
        self._ask: pd.Series | None = None
        self._mid_cache: dict[str, pd.Series] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        sub_id: str,
        instrument_id: str,
        market: str | None = None,
        currency: str | None = None,
    ) -> "LiveBook":
        """Register one subscription.

        Parameters
        ----------
        sub_id        : key used by market_data (e.g. "IM:IE00B4L5Y983").
        instrument_id : canonical security identifier (required).
        market        : optional market / venue metadata.
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

    # ── Filter configuration ──────────────────────────────────────────────────

    def add_filter(
        self,
        filt: BookFilter,
        securities: Sequence[str] | None = None,
    ) -> "LiveBook":
        """
        Attach a BookFilter.

        Parameters
        ----------
        filt       : A BookFilter (SpreadEWMA, PriceEWMA, …).
        securities : Sub IDs this filter applies to. None → global (all rows).

        Returns self for chaining.
        """
        scope = frozenset(securities) if securities is not None else None
        self._filters.append((filt, scope))
        return self

    # ── Update (hot path) ─────────────────────────────────────────────────────

    def update(self, raw: pd.DataFrame) -> None:
        """
        Ingest a BID/ASK DataFrame, apply filters, aggregate to instrument level.

        Parameters
        ----------
        raw : DataFrame with index=sub_id, columns including BID and/or ASK.
              Zeros are treated as NaN.
        """
        if self._instr_series is None:
            self._instr_series = pd.Series(self._sub_to_instr, dtype="object")

        self._mid_cache.clear()

        bid_ask_cols = [c for c in ("BID", "ASK") if c in raw.columns]
        if not bid_ask_cols:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        book = raw[bid_ask_cols].replace({0: np.nan}).dropna(how="all")

        if book.empty:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        # --- Apply filter chain -------------------------------------------
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
                outside = book.index.difference(scope)
                book = pd.concat([book.loc[outside], valid])

        if book.empty:
            self._bid = pd.Series(dtype=float)
            self._ask = pd.Series(dtype=float)
            return

        # --- Map sub_id → instrument_id (restrict to registered subs) ----
        known = self._instr_series.reindex(book.index).dropna()
        book = book.reindex(known.index)
        instr = known  # Series: sub_id → instrument_id

        # --- Vectorised groupby aggregation --------------------------------
        if "BID" in book.columns:
            self._bid = book["BID"].groupby(instr).max()
        else:
            self._bid = pd.Series(dtype=float)

        if "ASK" in book.columns:
            self._ask = book["ASK"].groupby(instr).min()
        else:
            self._ask = pd.Series(dtype=float)

        # Pre-compute all three methods so get_mid() is a dict lookup
        if "BID" in book.columns and "ASK" in book.columns:
            self._mid_cache["best_bid_ask"] = (self._bid + self._ask) / 2
            mean_bid = book["BID"].groupby(instr).mean()
            mean_ask = book["ASK"].groupby(instr).mean()
            self._mid_cache["mean_bid_ask"] = (mean_bid + mean_ask) / 2
            per_sub_mid = (book["BID"] + book["ASK"]) / 2
            self._mid_cache["mean_mids"] = per_sub_mid.groupby(instr).mean()
        elif "BID" in book.columns:
            self._mid_cache["best_bid_ask"] = self._bid.copy()
            self._mid_cache["mean_bid_ask"] = book["BID"].groupby(instr).mean()
            self._mid_cache["mean_mids"] = self._mid_cache["mean_bid_ask"]
        elif "ASK" in book.columns:
            self._mid_cache["best_bid_ask"] = self._ask.copy()
            self._mid_cache["mean_bid_ask"] = book["ASK"].groupby(instr).mean()
            self._mid_cache["mean_mids"] = self._mid_cache["mean_bid_ask"]

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_mid(self, method: str | None = None) -> pd.Series:
        """
        Return per-instrument mid prices.

        Parameters
        ----------
        method : "best_bid_ask" | "mean_bid_ask" | "mean_mids" | None
                 None uses the default_method set at construction.
        """
        m = method if method is not None else self._default_method
        if m not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {m!r}")
        return self._mid_cache.get(m, pd.Series(dtype=float))

    def get_bid_ask(self, instrument_id: str) -> tuple[float | None, float | None]:
        """Return (best_bid, best_ask) for a single instrument."""
        bid = self._bid.get(instrument_id) if self._bid is not None else None
        ask = self._ask.get(instrument_id) if self._ask is not None else None
        return (
            float(bid) if bid is not None and not np.isnan(bid) else None,
            float(ask) if ask is not None and not np.isnan(ask) else None,
        )
