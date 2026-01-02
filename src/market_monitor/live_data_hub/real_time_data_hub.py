"""
RTData - Refactored con API pulita e corretta

Correzioni:
- index_data -> store (naming consistente)
- Nessun accesso a membri privati degli store (_market_data, _last_update, etc.)
- Routing chiaro: "market", "state", "events", "blob" (NO "redis")
- Nessuna validazione valori (non è competenza di RTData)
- Currency handling ottimizzato
"""

import asyncio
import logging
import warnings
from datetime import datetime
from threading import Lock
from typing import Any, Union, List, Dict, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

from market_monitor.live_data_hub.data_store import (
    MarketStore,
    StateStore,
    EventStore,
    BlobStore
)
from market_monitor.live_data_hub.live_subscription import (
    SubscriptionGroup,
    LiveSubscription,
    RedisSubscription,
    BloombergSubscription,
    DataStore
)
from market_monitor.live_data_hub.subscription_service import SubscriptionService
from market_monitor.utils.decorators import deprecated


# Constants
SECURITY_TO_IGNORE = ["GBp", "OTHEREQUIEUR", "ILs"]
EUR_SYNONYM = ["EUREUR", "EUREUR CURNCY"]


def handle_currency_conversion(func):
    """Decorator to handle GBP/ILS conversion (GBP -> GBp, ILS -> ILs)"""

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

    Clean version with:
    - Proper encapsulation (no access to private store members)
    - Clear store routing (market/state/events/blob)
    - Simplified currency handling
    - No value validation (not RTData's responsibility)

    Example:
        >>> lock = Lock()
        >>> rtdata = RTData(lock, fields=["BID", "ASK"])
        >>> rtdata.set_securities(["ISIN1", "ISIN2"])
        >>> rtdata.currency_information = {"ISIN1": "USD", "ISIN2": "EUR"}
        >>> rtdata.update("ISIN1", {"BID": 100.0, "ASK": 102.0})
    """

    def __init__(self,
                 locker: Union[Lock, asyncio.Lock],
                 fields: Optional[List[str]] = None,
                 mid_key: Optional[Union[str, List[str]]] = None,
                 subscription_dict: Optional[Dict] = None,
                 instruments_status: Optional[Dict] = None,
                 currency_information: Optional[Dict[str, str]] = None,
                 subscription_service: Optional[SubscriptionService] = None,
                 **kwargs):
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

        # Subscription service
        self._subscription_service = subscription_service or SubscriptionService()

        # Currency information
        self._currency_information: Dict[str, str] = {}

        # Configuration
        self._securities: set = set()
        self.missing_currency_instruments: set = set()
        self.currencies_in_book: set = set()

        # Initialize subscriptions
        if subscription_dict:
            self._subscription_service.apply_subscription_dict(subscription_dict)

        # Initialize currency info
        if currency_information:
            self.currency_information = currency_information

        # Initialize instrument status
        if instruments_status:
            for isin, status in instruments_status.items():
                self._subscription_service.update_instrument_status(isin, status)

        # Backward compatibility
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
                       store: str = "market") -> None:
        """Set monitored securities and initialize storage"""
        normalized = self._subscription_service.set_allowed_securities(securities)

        # Update
        self._securities = set(normalized)
        self.subscription_status = {s: {} for s in self._securities}

        # Initialize storage
        if store == "market":
            self._market_store.initialize(list(self._securities))

    # ========================================================================
    # DATA UPDATES
    # ========================================================================

    def update(self, ticker: str, data: Dict[str, Any],
               store: Union[str, DataStore] = "market") -> None:
        """
        Update data for a ticker (thread-safe).

        Args:
            ticker: Security identifier
            data: Field updates (e.g., {"BID": 100.0, "ASK": 102.0})
            store: Target store - "market", "state", "events", or "blob"
        """
        # Normalize store
        if isinstance(store, str):
            try:
                store = DataStore(store)
            except ValueError:
                store = DataStore.MARKET

        if store == DataStore.MARKET:
            # Update market store
            self._market_store.update(ticker, data)

            # Track last update time
            now = datetime.now()
            self.last_update[ticker] = now
            if len(self.last_update) > self._max_last_update_size:
                self.last_update.popitem(last=False)

        elif store == DataStore.STATE:
            # Update state store (namespace = ticker)
            for key, value in data.items():
                self._state_store.update(ticker, key, value)

        elif store == DataStore.EVENTS:
            # Append to event store
            self._event_store.append(ticker, data)

        elif store == DataStore.BLOB:
            # Store as blob
            self._blob_store.store(ticker, data)

    def update_all_data(self, data: Union[List, pd.DataFrame, dict, object],
                        store: Optional[Union[str, List[str]]] = None) -> None:
        """
        Update data with multiple DataFrames or nested dicts.

        Args:
            data: Data to update (DataFrame, dict, or list of these)
            store: Target store(s) - "market", "state", "events", "blob"
        """
        if store is None:
            self.logger.warning("update_all_data with store=None")
            return

        # Normalize inputs
        if isinstance(data, (pd.DataFrame, dict)):
            data = [data]
        if isinstance(store, str):
            store = [store]

        # Update each source
        for store_name, d in zip(store, data):
            if store_name == "market" and isinstance(d, pd.DataFrame):
                self._market_store.set_dataframe(d)

            elif store_name == "state":
                if isinstance(d, (pd.Series, pd.DataFrame)):
                    d = d.to_dict()

                if isinstance(d, dict):
                    # Store in state (namespace per key)
                    for key, value in d.items():
                        if isinstance(value, dict):
                            # Nested dict: update each sub-key
                            for sub_key, sub_value in value.items():
                                self._state_store.update(key, sub_key, sub_value)
                        else:
                            # Simple value
                            self._state_store.update("data", key, value)
                else:
                    self._state_store.update("data", "value", d)

            elif store_name == "events":
                if isinstance(d, list):
                    for event in d:
                        self._event_store.append("default", event)
                else:
                    self._event_store.append("default", d)

            elif store_name == "blob":
                if isinstance(d, dict):
                    for key, value in d.items():
                        self._blob_store.store(key, value)

    # ========================================================================
    # DATA RETRIEVAL
    # ========================================================================

    def get_data_field(self,
                       field: Optional[Union[str, List[str]]] = None,
                       store: str = "market",
                       securities: Optional[Union[str, List[str]]] = None) -> Union[pd.DataFrame, pd.Series, dict, Any]:
        """Get data for specified field(s) and securities (thread-safe)"""
        if store == "market":
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

        elif store == "state":
            # Get from state store
            if securities is None:
                return self._state_store.get(field) if field else self._state_store.get_all()
            else:
                # Get specific securities from state
                if isinstance(securities, str):
                    data = self._state_store.get(securities)
                    if field and isinstance(data, dict):
                        return data.get(field)
                    return data
                else:
                    result = {}
                    for sec in securities:
                        data = self._state_store.get(sec)
                        if field and isinstance(data, dict):
                            result[sec] = data.get(field)
                        else:
                            result[sec] = data
                    return result

        elif store == "events":
            # Get from event store
            event_type = field or "default"
            return self._event_store.get(event_type)

        elif store == "blob":
            # Get from blob store
            if securities is None:
                return {key: self._blob_store.get(key) for key in self._blob_store.list_keys()}
            else:
                if isinstance(securities, str):
                    return self._blob_store.get(securities)
                else:
                    return {sec: self._blob_store.get(sec) for sec in securities}

        return None

    @handle_currency_conversion
    def get_mid(self, securities: Optional[Union[str, List[str]]] = None,
                store: str = "market") -> Union[pd.Series, float]:
        """Calculate and return mid-value (thread-safe, optimized)"""
        if store == "market":
            tickers = None
            if securities is not None:
                tickers = [securities] if isinstance(securities, str) else securities

            result = self._market_store.get_mid(self.mid_key, tickers)

            # Return float for single security
            if isinstance(securities, str):
                return result[securities] if securities in result.index else np.nan

            return result

        elif store == "state":
            # Calculate mid from state data
            data = self.get_data_field(field=None, store=store, securities=securities)

            if isinstance(data, dict) and securities:
                securities_list = [securities] if isinstance(securities, str) else securities
                result = {}

                for sec in securities_list:
                    if sec in data and isinstance(data[sec], dict):
                        values = [data[sec].get(f) for f in self.mid_key if f in data[sec]]
                        if values:
                            result[sec] = sum(values) / len(values)

                return pd.Series(result) if len(result) > 1 else result.get(securities, np.nan)

            return pd.Series()

        return pd.Series()

    def get_mid_eur(self, currencies: List[str],  store: str = "market") -> pd.Series:
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
            if ticker in currencies:
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
                # Check for zero FX rate to prevent division by zero
                if fx_rates[ccy] != 0:
                    converted[isin] = price / fx_rates[ccy]
                else:
                    converted[isin] = price
                    self.logger.warning(f"FX rate for {ccy} is zero, skipping conversion for {isin}")
            else:
                converted[isin] = price  # Fallback: no conversion

        # Invert FX rates (CCY/EUR -> EUR/CCY)
        fx_inverted = {ccy: 1.0 / rate for ccy, rate in fx_rates.items() if rate != 0}

        # Combine
        converted.update(fx_inverted)

        return pd.Series(converted)

    @handle_currency_conversion
    def get_book_eur(self, currencies: List[str],  store: str = "market") -> pd.DataFrame:
        """
        Get book converted to EUR (optimized).

        Uses public API of MarketStore instead of accessing private members.
        """
        # Get full market book
        book = self._market_store.get_data()

        if book.empty:
            return book

        # Identify instruments vs currencies
        is_currency = book.index.isin(currencies)

        # Split book
        book_instr = book[~is_currency]
        book_fx = book[is_currency]

        if book_instr.empty:
            return book

        # Calculate FX mid prices (using public get_mid)
        fx_mid_series = self._market_store.get_mid(["BID", "ASK"],
                                                    book_fx.index.tolist() if not book_fx.empty else None)
        fx_mid = fx_mid_series.to_dict()

        # Convert instruments to EUR
        book_instr_eur = book_instr.copy()
        for isin in book_instr.index:
            ccy = self._currency_information.get(isin, "EUR")
            if ccy != "EUR" and ccy in fx_mid and fx_mid[ccy] != 0:
                book_instr_eur.loc[isin] = book_instr.loc[isin] / fx_mid[ccy]

        # Invert FX book
        if not book_fx.empty:
            book_fx_inverted = 1.0 / book_fx
            return pd.concat([book_instr_eur, book_fx_inverted])

        return book_instr_eur

    def get_available_fields(self, store: Optional[str] = None) -> list[str] | dict[str, list[str]]:
        """Get available field names"""
        if store == "market":
            return self.fields
        elif store == "state":
            # Return namespaces from state store
            all_state = self._state_store.get_all()
            return list(all_state.keys())
        elif store == "events":
            # Return event types
            all_events = self._event_store.get_all()
            return list(all_events.keys())
        elif store == "blob":
            return self._blob_store.list_keys()
        else:
            # All stores
            return {
                "market": self.fields,
                "state": list(self._state_store.get_all().keys()),
                "events": list(self._event_store.get_all().keys()),
                "blob": self._blob_store.list_keys()
            }

    # ========================================================================
    # CURRENCY MANAGEMENT
    # ========================================================================

    @property
    def currency_information(self) -> Dict[str, str]:
        """
        Get currency information for all instruments.

        Returns:
            Dict mapping ISIN -> currency code (e.g., {"ISIN1": "USD"})
        """
        return self._currency_information.copy()

    @currency_information.setter
    def currency_information(self, value: Union[Dict[str, str], pd.Series]):
        """
        Set currency information for instruments.

        Args:
            value: Dict or Series mapping ISIN -> currency code

        Example:
            >>> rtdata = RTData()
            >>> rtdata.currency_information = {"ISIN1": "USD", "ISIN2": "EUR"}
        """
        # Convert Series to dict if needed
        if isinstance(value, pd.Series):
            value = value.to_dict()

        # Filter out empty keys/values
        value = {k: v for k, v in value.items() if k and v}

        # Validate and set currencies
        for isin, ccy in value.items():
            # Warn if instrument not in securities
            if isin not in self._securities:
                self.logger.debug(f"Setting currency for unknown instrument {isin}")


            self._currency_information[isin] = ccy

        # Ensure EUR maps to EUR
        self._currency_information["EUR"] = "EUR"
        self._currency_information["GBp"] = "EUR"

        # Add missing currencies to book
        all_currencies = set(value.values()) - {"EUR", "EUREUR", "EUREUR CURNCY", "GBp"}
        unsubscribed_currencies = all_currencies - self._securities

        # Add to market data using public API
        if unsubscribed_currencies:
            for ccy in unsubscribed_currencies:
                if not self._market_store.has_ticker(ccy):
                    # Initialize with zeros
                    init_data = {field: 0.0 for field in self.fields}
                    self._market_store.update(ccy, init_data)

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
    # SUBSCRIPTION MANAGEMENT (Deprecated - use get_subscription_manager())
    # ========================================================================
    #
    # These methods are deprecated. Use rtdata.get_subscription_manager() instead:
    #   rtdata.get_subscription_manager().subscribe_bloomberg(...)
    #   rtdata.get_subscription_manager().get_all_subscriptions()
    # ========================================================================



    @property
    def subscription_dict_bloomberg(self) -> Dict[str, str]:
        """Get Bloomberg subscription dictionary (Deprecated: use get_subscription_manager())"""
        warnings.warn(
            "subscription_dict_bloomberg is deprecated. Use rtdata.get_subscription_manager() instead.",
            FutureWarning, stacklevel=2
        )
        return self._subscription_service.get_all_subscriptions_dict().get("bloomberg", {})

    @property
    def subscription_dict_redis(self) -> List[str]:
        """Get Redis subscriptions (Deprecated: use get_subscription_manager())"""
        warnings.warn(
            "subscription_dict_redis is deprecated. Use rtdata.get_subscription_manager() instead.",
            FutureWarning, stacklevel=2
        )
        return self._subscription_service.get_all_subscriptions_dict().get("redis", [])


    @subscription_dict_bloomberg.setter
    def subscription_dict_bloomberg(self, value: Union[pd.Series, dict]):
        """Set Bloomberg subscriptions (Deprecated: use get_subscription_manager())"""
        warnings.warn(
            "subscription_dict_bloomberg setter is deprecated. Use rtdata.get_subscription_manager().set_bloomberg_subscriptions() instead.",
            FutureWarning, stacklevel=2
        )
        self._subscription_service.set_bloomberg_subscriptions(value)

    @subscription_dict_redis.setter
    def subscription_dict_redis(self, value: List[str]):
        """Set Redis subscriptions (Deprecated: use get_subscription_manager())"""
        warnings.warn(
            "subscription_dict_redis setter is deprecated. Use rtdata.get_subscription_manager().set_redis_subscriptions() instead.",
            FutureWarning, stacklevel=2
        )
        self._subscription_service.set_redis_subscriptions(value)

    @property
    @deprecated("Use same method of SubscriptionService Instead.")
    def subscription_dict(self) -> Dict[str, Any]:
        """Get all subscriptions (Deprecated: use get_subscription_manager())"""
        warnings.warn(
            "subscription_dict is deprecated. Use rtdata.get_subscription_manager().get_all_subscriptions_dict() instead.",
            FutureWarning, stacklevel=2
        )
        return self._subscription_service.get_all_subscriptions_dict()

    @subscription_dict.setter
    def subscription_dict(self, value: Dict[str, Any]):
        """Set subscriptions for multiple sources (Deprecated: use get_subscription_manager())"""
        warnings.warn(
            "subscription_dict setter is deprecated. Use rtdata.get_subscription_manager().apply_subscription_dict() instead.",
            FutureWarning, stacklevel=2
        )
        self._subscription_service.apply_subscription_dict(value)

    @property
    def subscription_source(self) -> Dict[str, Any]:
        """Get subscription sources (Deprecated: use get_subscription_manager())"""

        return self._subscription_service.get_all_subscriptions_dict()

    @deprecated("Use same method of SubscriptionService Instead.")
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
            group: Optional group to add subscription to

        Returns:
            BloombergSubscription instance
        """
        return self._subscription_service.subscribe_bloomberg(
            id=id,
            subscription_string=subscription_string,
            fields=fields or self.fields,
            params=params,
            group=group
        )

    @deprecated("Use same method of SubscriptionService Instead.")
    def subscribe_redis(self,
                        channel: str,
                        subscription: Optional[str] = None,
                        store: Optional[str] = None,
                        event_type: Optional[str] = None,
                        fields: Optional[List[str]] = None,
                        id: Optional[str] = None,
                        group: Optional[str] = None) -> RedisSubscription:
        """
        Subscribe to Redis channel with routing configuration.

        Args:
            channel: Redis channel name (e.g., "market:prices", "state:portfolio:cash")
            subscription: Subscription pattern (defaults to channel, supports wildcards)
            store: Target store - "market", "state", "events", or "blob"
            event_type: Custom event type name (ONLY for store="events")
            fields: Optional list of fields
            id: Optional unique ID
            group: Optional group to add subscription to

        Returns:
            RedisSubscription instance
        """
        return self._subscription_service.subscribe_redis(
            channel=channel,
            subscription=subscription,
            store=store,
            event_type=event_type,
            fields=fields,
            id=id,
            group=group
        )

    @deprecated("Use same method of SubscriptionService Instead.")
    def subscribe(self, id: str, source: str, subscription_string: str,
                  fields: Optional[List[str]] = None, group: Optional[str] = None) -> LiveSubscription:
        """
        Generic subscribe method (deprecated - use subscribe_bloomberg or subscribe_redis).
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

    @deprecated("Use same method of SubscriptionService Instead.")
    def unsubscribe(self, id: str, source: str) -> bool:
        """Mark subscription for removal"""
        return self._subscription_service.unsubscribe(id, source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_subscription(self, ticker: str, source: str) -> Optional[LiveSubscription]:
        """Get subscription details for a ticker"""
        return self._subscription_service.get_subscription(ticker, source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_bloomberg_subscription(self, ticker: Optional[str] = None) -> Optional[Dict[str, BloombergSubscription]]:
        """Get Bloomberg subscription for a ticker"""
        return self._subscription_service.get_bloomberg_subscription(ticker)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_redis_subscription(self, ticker: Optional[str] = None) -> Optional[Dict[str, RedisSubscription]]:
        """Get Redis subscription for a ticker"""
        return self._subscription_service.get_redis_subscription(ticker)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_all_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """Get all subscriptions"""
        return self._subscription_service.get_all_subscriptions(source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_active_subscriptions(self, source: Optional[str] = None) -> List[LiveSubscription]:
        """Get all active subscriptions"""
        return self._subscription_service.get_active_subscriptions(source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def create_subscription_group(self, name: str, metadata: Optional[Dict] = None) -> SubscriptionGroup:
        """Create a named subscription group"""
        return self._subscription_service.create_subscription_group(name, metadata)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_subscription_group(self, name: str) -> Optional[SubscriptionGroup]:
        """Get subscription group by name"""
        return self._subscription_service.get_subscription_group(name)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_subscription_health(self) -> Dict[str, Any]:
        """Get subscription health metrics"""
        return self._subscription_service.get_subscription_health()

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_failed_subscriptions(self) -> List[Dict[str, Any]]:
        """Get detailed report of failed subscriptions"""
        return self._subscription_service.get_failed_subscriptions()

    @deprecated("Use same method of SubscriptionService Instead.")
    def mark_subscription_received(self, id: str, source: str = "bloomberg"):
        """Mark that data was received for a subscription"""
        self._subscription_service.mark_subscription_received(id, source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def mark_subscription_failed(self, id: str, source: str, error: Optional[str] = None):
        """Mark subscription as failed"""
        self._subscription_service.mark_subscription_failed(id, source, error)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_pending_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """Get subscriptions waiting to be activated"""
        return self._subscription_service.get_pending_subscriptions(source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_to_unsubscribe(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """Get subscriptions marked for removal"""
        return self._subscription_service.get_to_unsubscribe(source)

    @deprecated("Use same method of SubscriptionService Instead.")
    def clear_unsubscribed(self, id: str, source: str):
        """Clear subscription from unsubscribe queue after processing"""
        self._subscription_service.clear_unsubscribed(id, source)

    @deprecated("Use same method of SubscriptionService Instead.")
    @property
    def instrument_status(self) -> Dict[str, str]:
        """Get instrument status"""
        return self._subscription_service.instrument_status

    def get_subscription_manager(self) -> SubscriptionService:
        """Accessor to the centralized SubscriptionService"""
        return self._subscription_service

    @deprecated("Use same method of SubscriptionService Instead.")
    def get_delayed_status(self) -> pd.Series:
        """Get delayed stream status for instruments"""
        return self._subscription_service.get_delayed_status()

