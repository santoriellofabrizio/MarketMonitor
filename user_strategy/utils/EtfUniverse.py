"""
EtfUniverse: lightweight ETF universe loader wrapping the oracle API.

Unifies the loading patterns of EtfEquityPriceEngine (single market, equity
underlying) and CreditPriceEngine (multi-market, fixed-income underlying) in
one reusable class.

Typical usage
-------------
Equity engine style (one market, equity)::

    universe = EtfUniverse(api, markets=["IM"], underlying="EQUITY")
    isins   = universe.isins
    tickers = universe.get_tickers()

Credit engine style (multi-market, FI)::

    universe = EtfUniverse(
        api,
        markets=["IM", "FP", "NA"],
        underlying=["FIXED INCOME", "MONEY MARKET"],
    )
    isins                   = universe.isins
    etfs_by_market          = universe.by_market
    currency_per_isin_mkt   = universe.currency_per_isin_market
    tickers                 = universe.get_tickers()
"""
from __future__ import annotations

from typing import Any, Sequence

from sfm_data_provider.interface.bshdata import BshData


class EtfUniverse:
    """
    ETF universe loaded from the oracle API.

    Parameters
    ----------
    api :
        BshData instance already initialised with the correct config.
    markets :
        Market segments to query, e.g. ``["IM"]`` (equity) or
        ``["IM", "FP", "NA"]`` (credit).  Default: ``("IM",)``.
    currency :
        Currency filter forwarded to oracle.  Default: ``"EUR"``.
    underlying :
        Asset-class filter — a string or a list of strings.
        Examples: ``"EQUITY"``, ``["FIXED INCOME", "MONEY MARKET"]``.
        ``None`` → no filter (all underlying types).
    extra_fields :
        Additional per-ISIN fields to retrieve per market from oracle.
        ``"CURRENCY"`` is included by default; it populates
        :attr:`currency_per_isin_market`.  Pass ``()`` to skip.
    """

    def __init__(
        self,
        api: BshData,
        markets: Sequence[str] = ("IM",),
        currency: str = "EUR",
        underlying: str | Sequence[str] | None = None,
        extra_fields: Sequence[str] = ("CURRENCY",),
    ) -> None:
        self.instruments_by_type = None
        self._api = api
        self._markets = list(markets)
        # Nested dict: {market: {isin: {field: value}}}
        self._raw: dict[str, dict[str, dict[str, Any]]] = {}
        self._load(currency, underlying, list(extra_fields))

    # ── Internal loader ───────────────────────────────────────────────────────

    def _load(
        self,
        currency: str,
        underlying: str | Sequence[str] | None,
        extra_fields: list[str],
    ) -> None:
        for mkt in self._markets:
            kwargs: dict[str, Any] = dict(
                fields=["etp_isins"],
                segments=[mkt],
                currency=currency,
                source="oracle",
            )
            if underlying is not None:
                kwargs["underlying"] = underlying
            if extra_fields:
                kwargs["extra_fields"] = extra_fields
            result = self._api.general.get(**kwargs)
            self._raw[mkt] = result.get("etp_isins", {})

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def markets(self) -> list[str]:
        """Markets that were queried."""
        return list(self._markets)

    @property
    def isins(self) -> list[str]:
        """All unique ISINs across all markets, sorted."""
        return sorted({isin for data in self._raw.values() for isin in data})

    @property
    def by_market(self) -> dict[str, list[str]]:
        """``{market: [isin, ...]}`` — ISINs grouped by market segment."""
        return {mkt: list(data.keys()) for mkt, data in self._raw.items()}

    @property
    def currency_per_isin_market(self) -> dict[tuple[str, str], str]:
        """
        ``{(isin, market): currency}`` built from the ``CURRENCY`` extra field.

        Only populated when ``"CURRENCY"`` was included in *extra_fields*.
        """
        return {
            (isin, mkt): fields["CURRENCY"]
            for mkt, data in self._raw.items()
            for isin, fields in data.items()
            if "CURRENCY" in fields
        }

    # ── Getters ───────────────────────────────────────────────────────────────

    def get(self, market: str | None = None) -> list[str]:
        """
        Return ISINs, optionally restricted to a single market.

        Parameters
        ----------
        market :
            Market segment key (e.g. ``"IM"``).
            ``None`` → all unique ISINs across all markets.

        Examples
        --------
        >>> universe.get()        # all ISINs
        >>> universe.get("IM")    # only ISINs listed on IM
        """
        if market is None:
            return self.isins
        return list(self._raw.get(market, {}).keys())

    def get_currency(self, isin: str, market: str, default: str = "EUR") -> str:
        """
        Currency for a single ``(isin, market)`` pair.

        Returns *default* when the pair is not found or ``CURRENCY`` was not
        fetched as an extra field.
        """
        return self._raw.get(market, {}).get(isin, {}).get("CURRENCY", default)

    def get_currencies(self) -> dict[tuple[str, str], str]:
        """Alias for :attr:`currency_per_isin_market`."""
        return self.currency_per_isin_market

    def get_tickers(self) -> dict[str, str]:
        """
        Fetch ``{isin: ticker}`` from the API info layer.

        Makes a separate network call each time — cache the result locally if
        it will be called multiple times.

        Returns an empty dict when the universe is empty.
        """
        all_isins = self.isins
        if not all_isins:
            return {}
        return self._api.info.get_etp_fields("TICKER", isin=all_isins)["TICKER"].to_dict()

    def get_field(
        self,
        field: str,
        market: str | None = None,
        isin: str | None = None,
    ) -> dict[str, Any]:
        """
        Return ``{isin: value}`` for any oracle extra field.

        When the same ISIN appears in multiple markets (and *market* is not
        restricted), the last market's value is kept.

        Parameters
        ----------
        field :
            Field name — must have been requested in *extra_fields* at
            construction time (e.g. ``"CURRENCY"``).
        market :
            Restrict output to a single market; ``None`` → all markets.
        isin :
            Restrict to a single ISIN; ``None`` → all.
        """
        result: dict[str, Any] = {}
        markets = [market] if market else self._markets
        for mkt in markets:
            for k, fields in self._raw.get(mkt, {}).items():
                if isin is not None and k != isin:
                    continue
                if field in fields:
                    result[k] = fields[field]
        return result

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of unique ISINs across all markets."""
        return len(self.isins)

    def __contains__(self, isin: str) -> bool:
        """``True`` if *isin* is present in any of the queried markets."""
        return any(isin in data for data in self._raw.values())

    def __repr__(self) -> str:
        counts = ", ".join(f"{mkt}={len(v)}" for mkt, v in self._raw.items())
        return f"EtfUniverse({counts})"
