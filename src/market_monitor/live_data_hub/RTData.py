"""
RTData - Ottimizzato con currency_information rivista e meno Pandas overhead

Ottimizzazioni:
- CurrencyConverter semplificato e più veloce
- Meno conversioni pandas inutili
- Dict operations dove possibile (più veloci di Series)
- currency_information property più chiaro
"""

import asyncio
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Union, List, Dict, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field as dataclass_field
from enum import Enum

import numpy as np
import pandas as pd

from market_monitor.live_data_hub.data_store import (
    MarketStore,
    StateStore,
    EventStore,
    BlobStore
)
from market_monitor.live_data_hub.live_subscription import SubscriptionGroup, LiveSubscription, RedisSubscription, \
    BloombergSubscription, DataStore, LiveSubscriptionManager
from market_monitor.utils.enums import CURRENCY

# Constants
SECURITY_TO_IGNORE = ["GBp", "OTHEREQUIEUR", "ILs"]
EUR_SYNONYM = ["EUREUR", "EUREUR CURNCY"]


def handle_currency_conversion(func):
    """Decorator to handle GBP/ILS conversion (GBP → GBp, ILS → ILs)"""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, dict):
            if "GBP" in result:
                result["GBp"] = result["GBP"] * 100
            if "ILS" in result:
                result["ILs"] = result["ILS"] * 100
        elif isinstance(result, pd.DataFrame):
            if "GBP" in result.index:
                result.loc["GBp"] = result.loc["GBP"] * 100
            if "ILS" in result.index:
                result.loc["ILs"] = result.loc["ILS"] * 100
        elif isinstance(result, pd.Series):
            if "GBP" in result.index:
                result["GBp"] = result["GBP"] * 100
            if "ILS" in result.index:
                result["ILs"] = result["ILS"] * 100

        return result

    return wrapper


class RTData:
    """
    Thread-safe real-time market data manager.

    Optimized version with:
    - Simplified currency handling (no separate CurrencyConverter class)
    - Reduced pandas overhead (dict operations where faster)
    - Clear currency_information property

    Example:
        >>> lock = Lock()
        >>> rtdata = RTData(lock, fields=["BID", "ASK"])
        >>> rtdata.set_securities(["ISIN1", "ISIN2"])
        >>> rtdata.currency_information = {"ISIN1": "USD", "ISIN2": "EUR"}
        >>> rtdata.update("ISIN1",{"BID": 100.0, "ASK": 102.0})
    """

    def __init__(self,
                 locker: Union[Lock, asyncio.Lock],
                 fields: Optional[List[str]] = None,
                 mid_key: Optional[Union[str, List[str]]] = None,
                 max_var_threshold: Optional[float] = None,
                 subscription_dict: Optional[Dict] = None,
                 instruments_status: Optional[Dict] = None,
                 currency_information: Optional[Dict[str, str]] = None):
        """Initialize RTData"""
        self.logger = logging.getLogger(__name__)
        self.locker = locker

        # Fields configuration
        self.fields = fields or ["BID", "ASK"]
        self.mid_key = mid_key or self.fields
        if isinstance(self.mid_key, str):
            self.mid_key = [self.mid_key]

        # Data Stores (organized by data type)
        self._market_store = MarketStore(locker, self.fields)
        self._state_store = StateStore(locker)
        self._event_store = EventStore(locker, default_maxlen=1000)
        self._blob_store = BlobStore(locker)
        
        # Subscription manager
        self._subscriptions = LiveSubscriptionManager()

        # Currency information (kept in RTData for now)
        self._currency_information: Dict[str, str] = {}

        # Configuration
        self._max_var_threshold = max_var_threshold
        self._securities: set = set()
        self.threshold_exceeded_instr: set = set()
        self.missing_currency_instruments: set = set()
        self.currencies_in_book: set = set()

        # Initialize subscriptions
        if subscription_dict:
            for source, subs in subscription_dict.items():
                if source.lower() == "bloomberg" and isinstance(subs, dict):
                    self._subscriptions.set_bloomberg_subscriptions(subs)
                elif source.lower() == "redis":
                    self._subscriptions.set_redis_subscriptions(subs)

        # Initialize currency info
        if currency_information:
            self.currency_information = currency_information

        # Initialize instrument status
        if instruments_status:
            for isin, status in instruments_status.items():
                self._subscriptions.update_instrument_status(isin, status)

        # Backward compatibility
        from collections import OrderedDict
        self.last_update = OrderedDict()
        self._max_last_update_size = 10000
        self.subscription_status = {}

    # ========================================================================
    # SECURITIES MANAGEMENT
    # ========================================================================

    @property
    def securities(self) -> List[str]:
        """Get monitored securities (excluding EUR synonyms)"""
        return [s for s in self._securities
                if s.upper() not in ["EUR", "EUREUR", "EUREUR CURNCY"]]

    def set_securities(self, securities: Union[str, List[str]],
                       index_data: str = "market") -> None:
        """Set monitored securities and initialize storage"""
        if isinstance(securities, str):
            securities = [securities]

        # FIX: Filter out empty/None values
        securities = [s for s in securities if s]

        # Remove EUR synonyms
        for synonym in EUR_SYNONYM:
            while synonym in securities:
                self.logger.warning(f"Use 'EUR' instead of {synonym}")
                securities.remove(synonym)

        # Remove ignored securities
        for ignored in SECURITY_TO_IGNORE:
            while ignored in securities:
                securities.remove(ignored)

        # Update
        self._securities = set(securities)
        self.subscription_status = {s: {} for s in self._securities}

        # Initialize storage
        if index_data == "market":
            self._market_store.initialize(list(self._securities))

    # ========================================================================
    # DATA UPDATES
    # ========================================================================

    def update(self, ticker: str, data: Dict[str, Any],
               store: Union[str, DataStore] = "market",
               perform_check: bool = False) -> None:
        """
        Update market data for a ticker (thread-safe, optimized).

        FIX: When perform_check=True, old_values are acquired and update
        is performed in the SAME lock to prevent race conditions.
        """
        # Normalize source
        if isinstance(store, str):
            try:
                store = DataStore(store)
            except ValueError:
                store = DataStore.MARKET

        if store == DataStore.MARKET:
            if perform_check:
                # Get old values and update in same lock
                with self.locker:
                    old_values = {}
                    for field in data.keys():
                        try:
                            old_values[field] = self._market_store._market_data.at[ticker, field]
                        except:
                            old_values[field] = None

                    # Validate and update
                    rejected = []
                    for field, value in data.items():
                        old_val = old_values.get(field)
                        if old_val is not None:
                            if not self._market_store._is_valid_update(old_val, value, self._max_var_threshold):
                                rejected.append(field)
                                continue

                        self._market_store._market_data.at[ticker, field] = value

                    success = len(rejected) == 0
                    if success or (len(data) > len(rejected)):
                        now = datetime.now()
                        self._market_store._last_update[ticker] = now
                        self.last_update[ticker] = now
                        if len(self.last_update) > self._max_last_update_size:
                            self.last_update.popitem(last=False)

                if rejected:
                    self.logger.info(f"Rejected {len(rejected)} field(s) for {ticker}: exceeded threshold")
                    self.threshold_exceeded_instr.add(ticker)
            else:
                # No validation - simple update
                self._market_store.update(ticker, data)
                now = datetime.now()
                self.last_update[ticker] = now
                if len(self.last_update) > self._max_last_update_size:
                    self.last_update.popitem(last=False)

        elif store == DataStore.REDIS:
            # RedisPublisher data could go to state or market depending on context
            # For now, maintain backward compatibility - store in state
            self._state_store.update("redis", ticker, data)

    def update_all_data(self, data: Union[List, pd.DataFrame, dict, object],
                        index_data: Optional[Union[str, List[str]]] = None) -> None:
        """Update market data with multiple DataFrames or nested dicts"""
        if index_data is None:
            self.logger.warning("update_all_data with index_data=None")
            return

        # Normalize inputs
        if isinstance(data, (pd.DataFrame, dict)):
            data = [data]
        if isinstance(index_data, str):
            index_data = [index_data]

        # Update each source
        for idx, d in zip(index_data, data):
            if idx == "market" and isinstance(d, pd.DataFrame):
                self._market_store.set_dataframe(d)

            elif idx == "redis" or idx == "state":
                if isinstance(d, (pd.Series, pd.DataFrame)):
                    d = d.to_dict()

                if isinstance(d, dict):
                    # Store in state under the index name
                    for key, value in d.items():
                        self._state_store.update(idx, key, value)
                else:
                    self._state_store.update(idx, "data", d)

    # ========================================================================
    # DATA RETRIEVAL
    # ========================================================================

    def get_data_field(self,
                       field: Optional[Union[str, List[str]]] = None,
                       index_data: str = "market",
                       securities: Optional[Union[str, List[str]]] = None) -> Union[pd.DataFrame, pd.Series, dict, Any]:
        """Get data for specified field(s) and securities (thread-safe)"""
        if index_data == "market":
            if field is None:
                tickers = None
                if securities is not None:
                    tickers = [securities] if isinstance(securities, str) else securities
                return self._market_store.get_data(tickers=tickers)

            fields = [field] if isinstance(field, str) else field
            tickers = [securities] if isinstance(securities, str) else securities

            if len(fields) == 1:
                return self._market_store.get_field(fields[0], tickers)
            else:
                return self._market_store.get_data(tickers, fields)

        else:  # State or RedisPublisher
            # Get from state store
            if securities is None:
                return self._state_store.get(index_data, field)
            else:
                # Get specific securities from state
                data = self._state_store.get(index_data)
                if isinstance(securities, str):
                    return data.get(securities, {}).get(field) if field else data.get(securities)
                else:
                    result = {}
                    for sec in securities:
                        if sec in data:
                            result[sec] = data[sec].get(field) if field else data[sec]
                    return result

    def get_redis_channel_data(self, channel: str) -> Union[pd.DataFrame, pd.Series, dict, Any]:
        """Get data for specified RedisPublisher channel (from state store)"""
        return self._state_store.get("redis", channel)

    @handle_currency_conversion
    def get_mid(self, securities: Optional[Union[str, List[str]]] = None,
                index_data: str = "market") -> Union[pd.Series, float]:
        """Calculate and return mid-value (thread-safe, optimized)"""
        if index_data == "market":
            tickers = None
            if securities is not None:
                tickers = [securities] if isinstance(securities, str) else securities

            result = self._market_store.get_mid(self.mid_key, tickers)

            # Return float for single security
            if isinstance(securities, str):
                return result[securities] if securities in result.index else np.nan

            return result

        else:  # RedisPublisher
            data = self.get_data_field(field=None, index_data=index_data, securities=securities)

            if isinstance(data, dict) and securities:
                securities_list = [securities] if isinstance(securities, str) else securities
                result = {}

                for sec in securities_list:
                    if sec in data and isinstance(data[sec], dict):
                        values = [data[sec].get(f) for f in self.mid_key if f in data[sec]]
                        if values:
                            result[sec] = sum(values) / len(values)

                return pd.Series(result)

            return pd.Series()

    def get_mid_eur(self, index_data: str = "market") -> pd.Series:
        """
        Get mid prices converted to EUR (optimized with dict operations).

        Optimizations:
        - Uses dict for FX lookups (faster than Series)
        - Minimizes pandas operations
        - Vectorized numpy operations where possible
        """
        # Get mid prices as dict (faster for individual lookups)
        mid_dict = self._market_store.get_mid_dict(self.mid_key)

        if not mid_dict:
            return pd.Series()

        # Separate instruments from FX
        instruments = {}
        fx_rates = {}

        for ticker, price in mid_dict.items():
            if ticker in CURRENCY | {'GBp', 'ILs'}:
                fx_rates[ticker] = price
            else:
                instruments[ticker] = price

        if not instruments:
            return pd.Series(mid_dict)

        # Convert instruments to EUR (vectorized)
        converted = {}
        for isin, price in instruments.items():
            ccy = self._currency_information.get(isin, "EUR")
            if ccy == "EUR":
                converted[isin] = price
            elif ccy in fx_rates:
                # FIX: Check for zero FX rate to prevent division by zero
                if fx_rates[ccy] != 0:
                    converted[isin] = price / fx_rates[ccy]
                else:
                    # Fallback: no conversion
                    converted[isin] = price
                    self.logger.warning(f"FX rate for {ccy} is zero, skipping conversion for {isin}")
            else:
                converted[isin] = price  # Fallback: no conversion

        # Invert FX rates (CCY/EUR → EUR/CCY)
        fx_inverted = {ccy: 1.0 / rate for ccy, rate in fx_rates.items() if rate != 0}

        # Combine
        converted.update(fx_inverted)

        return pd.Series(converted)

    @handle_currency_conversion
    def get_book_eur(self, index_data: str = "market") -> pd.DataFrame:
        """
        Get book converted to EUR (optimized).

        Optimizations:
        - Single lock acquisition
        - Numpy operations for division
        - Minimal pandas overhead
        """
        with self.locker:
            book = self._market_store._market_data.copy()

            if book.empty:
                return book

            # Identify instruments vs currencies
            is_currency = book.index.isin(CURRENCY | {'GBp', 'ILs'})

            # Split book
            book_instr = book[~is_currency]
            book_fx = book[is_currency]

            if book_instr.empty:
                return book

            # Calculate FX mid (numpy is faster than pandas .mean())
            if not book_fx.empty:
                fx_mid_values = book_fx[["BID", "ASK"]].values.mean(axis=1)
                fx_mid = dict(zip(book_fx.index, fx_mid_values))
            else:
                fx_mid = {}

            # Convert instruments (vectorized numpy operation)
            book_instr_values = book_instr.values.copy()

            for i, isin in enumerate(book_instr.index):
                ccy = self._currency_information.get(isin, "EUR")
                if ccy != "EUR" and ccy in fx_mid:
                    book_instr_values[i] /= fx_mid[ccy]

            book_instr_eur = pd.DataFrame(
                book_instr_values,
                index=book_instr.index,
                columns=book_instr.columns
            )

            # Invert FX book
            if not book_fx.empty:
                book_fx_inverted = 1.0 / book_fx
                return pd.concat([book_instr_eur, book_fx_inverted])

            return book_instr_eur

    def get_available_fields(self, index_data: Optional[str] = None) -> List[str]:
        """Get available field names"""
        if index_data == "market":
            return self.fields
        else:
            # Return namespaces from state store
            all_state = self._state_store.get_all()
            return list(all_state.keys())

    # ========================================================================
    # CURRENCY MANAGEMENT (SIMPLIFIED)
    # ========================================================================

    @property
    def currency_information(self) -> Dict[str, str]:
        """
        Get currency information for all instruments.

        Returns:
            Dict mapping ISIN → currency code (e.g., {"ISIN1": "USD"})
        """
        return self._currency_information.copy()

    @currency_information.setter
    def currency_information(self, value: Union[Dict[str, str], pd.Series]):
        """
        Set currency information for instruments.

        Args:
            value: Dict or Series mapping ISIN → currency code

        Example:
            >>> rtdata = RTData()
            >>> rtdata.currency_information = {"ISIN1": "USD", "ISIN2": "EUR"}
            >>> # Or with pandas Series
            >>> rtdata.currency_information = pd.Series({"ISIN1": "USD"})

        Notes:
            - Unknown currencies default to EUR
            - Missing instruments get EUR
            - Automatically adds currency securities to book
        """
        # Convert Series to dict if needed
        if isinstance(value, pd.Series):
            value = value.to_dict()

        # FIX: Filter out empty keys/values
        value = {k: v for k, v in value.items() if k and v}

        # Validate and set currencies
        for isin, ccy in value.items():
            # Warn if instrument not in securities
            if isin not in self._securities:
                self.logger.debug(f"Setting currency for unknown instrument {isin}")

            # Validate currency
            if ccy not in CURRENCY | {'GBp', 'ILs', 'EUR'}:
                self.logger.warning(f"Unknown currency {ccy} for {isin}, defaulting to EUR")
                self.missing_currency_instruments.add(isin)
                self._currency_information[isin] = "EUR"
            else:
                self._currency_information[isin] = ccy

        # Ensure EUR maps to EUR
        self._currency_information["EUR"] = "EUR"
        self._currency_information["GBp"] = "EUR"

        # Add missing currencies to book
        all_currencies = set(value.values()) - {"EUR", "EUREUR", "EUREUR CURNCY", "GBp"}
        unsubscribed_currencies = all_currencies - self._securities

        # Add to market data (thread-safe)
        if unsubscribed_currencies:
            with self.locker:
                for ccy in unsubscribed_currencies:
                    if ccy not in self._market_store._market_data.index:
                        # Must provide values for all columns (fields)
                        self._market_store._market_data.loc[ccy] = [0.0] * len(self._market_store._fields)

        # Update tracking
        self.currencies_in_book = all_currencies | {"EUR"}

        # Add to securities set
        for ccy in self.currencies_in_book:
            if ccy not in self._securities:
                self._securities.add(ccy)

    def get_instruments_with_missing_ccy(self) -> List[str]:
        """Get list of instruments with missing/unknown currencies"""
        return list(self.missing_currency_instruments)

    # ========================================================================
    # SUBSCRIPTION MANAGEMENT
    # ========================================================================

    @property
    def subscription_dict_bloomberg(self) -> Dict[str, str]:
        """Get Bloomberg subscription dictionary"""
        return self._subscriptions.get_bloomberg_subscriptions()

    @subscription_dict_bloomberg.setter
    def subscription_dict_bloomberg(self, value: Union[pd.Series, dict]):
        """Set Bloomberg subscriptions"""
        self._subscriptions.set_bloomberg_subscriptions(value)

    @property
    def subscription_dict_redis(self) -> List[str]:
        """Get RedisPublisher subscriptions"""
        return self._subscriptions.get_redis_subscriptions()

    @subscription_dict_redis.setter
    def subscription_dict_redis(self, value: List[str]):
        """Set RedisPublisher subscriptions"""
        self._subscriptions.set_redis_subscriptions(value)

    @property
    def subscription_dict(self) -> Dict[str, Any]:
        """Get all subscriptions"""
        return self._subscriptions.get_all_subscriptions()

    @subscription_dict.setter
    def subscription_dict(self, value: Dict[str, Any]):
        """Set subscriptions for multiple sources"""
        for source, subs in value.items():
            if source.lower() == "bloomberg":
                self._subscriptions.set_bloomberg_subscriptions(subs)
            elif source.lower() == "redis":
                self._subscriptions.set_redis_subscriptions(subs)
            else:
                self.logger.warning(f"Unknown subscription source: {source}")

    @property
    def subscription_source(self) -> Dict[str, Any]:
        """Get subscription sources"""
        return self._subscriptions.get_all_subscriptions()

    def subscribe_bloomberg(self,
                            id: str,
                            subscription_string: Optional[str] = None,
                            fields: Optional[List[str]] = None,
                            params: Optional[Dict[str, Any]] = None,
                            group: Optional[str] = None) -> BloombergSubscription:
        """
        Subscribe to Bloomberg market data.

        Args:
            id: Security identifier (e.g., ISIN, ticker)
            subscription_string: Bloomberg subscription string (e.g., "AAPL US Equity")
            fields: List of Bloomberg fields (e.g., ["BID", "ASK", "LAST"])
            params: Additional Bloomberg parameters (e.g., {"interval": 1})
            id: Optional unique ID (auto-generated if not provided)
            group: Optional group to add subscription to

        Returns:
            BloombergSubscription instance

        Example:
            >>> rtdata = RTData()
            >>> sub = rtdata.subscribe_bloomberg(
            ...     ticker="US0378331005",
            ...     subscription_string="AAPL US Equity",
            ...     fields=["BID", "ASK", "LAST", "VOLUME"],
            ...     params={"interval": 1},
            ...     group="us_equities"
            ... )
        """
        bloomberg_sub = BloombergSubscription(
            id=id,
            subscription_string=subscription_string or id,
            fields=fields or self.fields,
            params=params or {}
        )
        self._subscriptions.add_subscription(bloomberg_sub, group=group)
        return bloomberg_sub

    @property
    def instrument_status(self) -> Dict[str, str]:
        """Get instrument status"""
        return self._subscriptions.get_instrument_status()

    # ========================================================================
    # NEW: Advanced Subscription API
    # ========================================================================

    def get_delayed_status(self) -> pd.Series:
        """Get delayed stream status for instruments"""
        return self._subscriptions.get_delayed_status()

    def subscribe_redis(self,
                        channel: str,
                        subscription: Optional[str] = None,
                        store: Optional[str] = None,
                        event_type: Optional[str] = None,
                        fields: Optional[List[str]] = None,
                        id: Optional[str] = None,
                        group: Optional[str] = None) -> RedisSubscription:
        """
        Subscribe to RedisPublisher channel with routing configuration.

        Args:
            channel: RedisPublisher channel name (e.g., "market:prices", "state:portfolio:cash")
            subscription: Subscription pattern (defaults to channel, supports wildcards like "market:*")
            store: Target store - "market", "state", "events", or "blob"
                   If None: auto-detect from channel (primo segmento se keyword, altrimenti default="market")
            event_type: Custom event type name (ONLY for store="events")
                       If None with store="events": usa channel come event type
            fields: Optional list of fields (per compatibilità)
            id: Optional unique ID (auto-generated if not provided)
            group: Optional group to add subscription to

        Returns:
            RedisSubscription instance

        Raises:
            ValueError: If event_type is provided but store != "events"

        Examples:
            # Market - explicit store
            >>> rtdata.subscribe_redis(
            ...     channel="prices",
            ...     subscription="prices:*",
            ...     store="market"
            ... )

            # Market - self-describing
            >>> rtdata.subscribe_redis(
            ...     channel="market:EUR:EONIA",
            ...     subscription="market:*"
            ... )  # store auto-detect → "market"

            # State - self-describing
            >>> rtdata.subscribe_redis(
            ...     channel="state:portfolio:cash",
            ...     subscription="state:*"
            ... )  # store auto-detect → "state", path = ["portfolio", "cash"]

            # Events - default event_type
            >>> rtdata.subscribe_redis(
            ...     channel="events:trades",
            ...     store="events"
            ... )  # event_type = "events:trades" (dal channel)

            # Events - custom event_type
            >>> rtdata.subscribe_redis(
            ...     channel="events:trades:executed",
            ...     store="events",
            ...     event_type="my_trades"
            ... )  # ← Custom

            # ❌ ERROR: event_type con store diverso
            >>> rtdata.subscribe_redis(
            ...     channel="market:prices",
            ...     store="market",
            ...     event_type="trades"  # ← ValueError!
            ... )
        """
        redis_sub = RedisSubscription(
            id=id or f"{channel.replace(':', '_')}",
            channel=channel,
            subscription=subscription or channel,
            store=store,
            event_type=event_type,
            fields=fields
        )
        self._subscriptions.add_subscription(redis_sub, group=group)
        return redis_sub

    def subscribe(self, id: str, source: str, subscription_string: str,
                  fields: Optional[List[str]] = None, group: Optional[str] = None) -> LiveSubscription:
        """
        Generic subscribe method (deprecated - use subscribe_bloomberg or subscribe_redis).

        This method is kept for backward compatibility but it's recommended to use
        the source-specific methods instead.
        """
        if source.lower() == "bloomberg":
            return self.subscribe_bloomberg(
                id=id,
                subscription_string=subscription_string,
                fields=fields,
                group=group
            )
        elif source.lower() == "redis":
            return self.subscribe_redis(
                id=id,
                channel=subscription_string,
                fields=fields,
                group=group
            )
        else:
            raise ValueError(f"Unknown source: {source}. Use 'bloomberg' or 'redis'")

    def unsubscribe(self, id: str, source: str) -> bool:
        """
        Mark subscription for removal.

        The subscription will be removed by BloombergThread on next check.

        Args:
            id: Security identifier
            source: Data source

        Returns:
            True if marked for unsubscribe, False otherwise
        """
        return self._subscriptions.mark_for_unsubscribe(id, source)

    def get_subscription(self, ticker: str, source: str) -> Optional[LiveSubscription]:
        """Get subscription details for a ticker"""
        return self._subscriptions.get_subscription(ticker, source)

    def get_bloomberg_subscription(self, ticker: Optional[str] = None) -> Optional[Dict[str, BloombergSubscription]]:
        """
        Get Bloomberg subscription for a ticker.

        Args:
            ticker: Security identifier

        Returns:
            BloombergSubscription or None if not found
        """
        if ticker:
            sub = self._subscriptions.get_subscription(ticker, "bloomberg")
            if isinstance(sub, BloombergSubscription):
                return {ticker: sub}
        else:
            return self._subscriptions.get_subscriptions_by_source("bloomberg")
        return None

    def get_redis_subscription(self, ticker: Optional[str] = None) -> Optional[Dict[str, RedisSubscription]]:
        """
        Get RedisPublisher subscription for a ticker.

        Args:
            ticker: Ticker identifier

        Returns:
            RedisSubscription or None if not found
        """
        if ticker:
            sub = self._subscriptions.get_subscription(ticker, "redis")
            if isinstance(sub, RedisSubscription):
                return {ticker: sub}
            return None
        return self._subscriptions.get_subscriptions_by_source("redis")

    def get_all_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """
        Get all subscriptions.

        Args:
            source: Filter by source (None = all sources)

        Returns:
            Dict of ticker → LiveSubscription
        """
        return self._subscriptions.get_all_live_subscriptions(source)

    def get_active_subscriptions(self, source: Optional[str] = None) -> List[LiveSubscription]:
        """Get all active subscriptions"""
        return self._subscriptions.get_active_subscriptions(source)

    def create_subscription_group(self, name: str, metadata: Optional[Dict] = None) -> SubscriptionGroup:
        """
        Create a named subscription group.

        Useful for organizing subscriptions by category (equities, bonds, etc).

        Args:
            name: Group name
            metadata: Optional metadata

        Returns:
            SubscriptionGroup instance
        """
        return self._subscriptions.create_group(name, metadata)

    def get_subscription_group(self, name: str) -> Optional[SubscriptionGroup]:
        """Get subscription group by name"""
        return self._subscriptions.get_group(name)

    def get_subscription_health(self) -> Dict[str, Any]:
        """
        Get subscription health metrics.

        Returns:
            Dict with counts of active, failed, stale subscriptions

        Example:
            >>> rt_data = RTData()
            >>> health = rt_data.get_subscription_health()
            >>> print(f"Active: {health['active']}, Failed: {health['failed']}")
        """
        return self._subscriptions.get_subscription_health()

    def get_failed_subscriptions(self) -> List[Dict[str, Any]]:
        """
        Get detailed report of failed subscriptions.

        Returns:
            List of dicts with failure details
        """
        return self._subscriptions.get_failed_subscriptions_report()

    def mark_subscription_received(self, id: str, source: str = "bloomberg"):
        """
        Mark that data was received for a subscription.

        This updates last_update time and ACTIVATES pending subscriptions.

        Args:
            id: Subscription identifier
            source: Data source
        """
        # Try to activate if pending
        activated = self._subscriptions.activate_subscription(id, source)

        if not activated:
            # Already active, just update
            self._subscriptions.mark_update_received(id, source)

    def mark_subscription_failed(self, id: str, source: str, error: Optional[str] = None):
        """
        Mark subscription as failed.

        Moves from pending/active to failed state.

        Args:
            id: Subscription identifier
            source: Data source
            error: Error message
        """
        self._subscriptions.fail_subscription(id, source, error or "Unknown error")

    def get_pending_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """
        Get subscriptions waiting to be activated.

        Args:
            source: Filter by source (None = all sources)

        Returns:
            Dict of id → LiveSubscription (pending)
        """
        return self._subscriptions.get_pending_subscriptions(source)

    def get_to_unsubscribe(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """
        Get subscriptions marked for removal.

        Args:
            source: Filter by source (None = all sources)

        Returns:
            Dict of id → LiveSubscription (to unsubscribe)
        """
        return self._subscriptions.get_to_unsubscribe(source)

    def clear_unsubscribed(self, id: str, source: str):
        """
        Clear subscription from unsubscribe queue after processing.

        Called by BloombergThread after successful unsubscribe.

        Args:
            id: Subscription identifier
            source: Data source
        """
        self._subscriptions.clear_unsubscribed(id, source)

