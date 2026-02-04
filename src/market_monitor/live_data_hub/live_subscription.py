import logging
from datetime import datetime
from typing import Any, Union, List, Dict, Optional
from dataclasses import dataclass, field as dataclass_field
from enum import Enum

import pandas as pd


class DataStore(str, Enum):
    """Data sources for market data"""
    MARKET = "market"
    STATE = "state"
    EVENTS = "events"
    BLOB = "blob"

    def __str__(self):
        return self.value


@dataclass
class LiveSubscription:
    """
    Base class for live market data subscriptions.

    Tracks subscription lifecycle and health.
    """
    id: str
    source: Optional[str] = None
    status: str = "pending"  # pending -> active -> failed/closed
    is_delayed: bool = False
    last_update: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def mark_active(self):
        """Mark subscription as active"""
        self.status = "active"
        self.error_count = 0

    def mark_failed(self, error: str):
        """Mark subscription as failed"""
        self.status = "failed"
        self.error_count += 1
        self.metadata["last_error"] = error
        self.metadata["last_error_time"] = datetime.now()

    def mark_closed(self):
        """Mark subscription as closed"""
        self.status = "closed"

    def update_received(self):
        """Called when data is received"""
        self.last_update = datetime.now()
        if self.status == "pending":
            self.mark_active()

    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if subscription hasn't received data recently"""
        if self.last_update is None:
            return False
        return (datetime.now() - self.last_update).total_seconds() > timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source": self.source,
            "status": self.status,
            "is_delayed": self.is_delayed,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "error_count": self.error_count,
            "metadata": self.metadata
        }


@dataclass
class BloombergSubscription(LiveSubscription):
    """
    Bloomberg-specific subscription.

    Attributes:
        id: Unique identifier for the subscription
        subscription_string: Bloomberg subscription string (e.g., "AAPL US Equity")
        fields: List of Bloomberg fields to subscribe to (e.g., ["BID", "ASK", "LAST"])
        params: Additional Bloomberg parameters (e.g., {"interval": 5})

    Example:
        >>> sub = BloombergSubscription(
        ...     id="sub_001",
        ...     ticker="US0378331005",
        ...     subscription_string="AAPL US Equity",
        ...     fields=["BID", "ASK", "LAST", "VOLUME"],
        ...     params={"interval": 1}
        ... )
    """
    id: str = ""
    subscription_string: str = ""
    fields: List[str] = dataclass_field(default_factory=list)
    params: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Initialize source after dataclass creation"""
        self.source = "bloomberg"
        if not self.id:
            # Generate ID from ticker if not provided
            self.id = self.subscription_string

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Bloomberg-specific fields"""
        base = super().to_dict()
        base.update({
            "id": self.id,
            "subscription_string": self.subscription_string,
            "fields": self.fields,
            "params": self.params
        })
        return base

    def get_bloomberg_request(self) -> Dict[str, Any]:
        """
        Get Bloomberg API request format.

        Returns:
            Dict ready for Bloomberg API
        """
        return {
            "securities": [self.subscription_string],
            "fields": self.fields,
            "options": self.params
        }


@dataclass
class RedisSubscription(LiveSubscription):
    """RedisPublisher subscription con routing dichiarativo."""
    channel: str = ""
    id: str = ""
    subscription: str = ""
    store: Optional[str] = None
    event_type: Optional[str] = None  # Solo per store="events"
    fields: Optional[List[str]] = None

    def __post_init__(self):
        self.source = "redis"
        if not self.id:
            self.id = f"redis_{self.channel.replace(':', '_')}"
        if not self.subscription:
            self.subscription = self.channel
        if not self.channel:
            raise ValueError("channel is required for redis subscriptions")

        # VALIDAZIONE: event_type solo per events store
        if self.event_type is not None and self.store != "events":
            raise ValueError(
                f"event_type can only be used with store='events'. "
                f"Got store='{self.store}' with event_type='{self.event_type}'"
            )


@dataclass
class KafkaSubscription(LiveSubscription):
    """
    Kafka subscription per dati di mercato real-time (DUMA, Binance, etc.).
    
    I messaggi Kafka sono in formato Avro e contengono dati per TUTTI gli strumenti.
    Il filtraggio avviene client-side tramite symbol_filter.
    
    Attributes:
        id: Identificatore univoco (es. ISIN o ticker)
        topic: Topic Kafka (es. "COALESCENT_DUMA.ETFP.BookBest")
        symbol_filter: Valore per filtrare i messaggi (es. ISIN "IE00B4L5Y983")
        symbol_field: Path del campo nel messaggio per il filtro (es. "instrument.isin")
        store: Target store - "market", "state", "events", "blob"
        fields_mapping: Mapping target_field -> source_path (supporta nested paths)
        
    Example:
        >>> sub = KafkaSubscription(
        ...     id="IWDA",
        ...     topic="COALESCENT_DUMA.ETFP.BookBest",
        ...     symbol_filter="IE00B4L5Y983",
        ...     symbol_field="instrument.isin",
        ...     store="market",
        ...     fields_mapping={
        ...         "BID": "bidBestLevel.price",
        ...         "ASK": "askBestLevel.price",
        ...         "BID_SIZE": "bidBestLevel.quantity",
        ...         "ASK_SIZE": "askBestLevel.quantity"
        ...     }
        ... )
    """
    id: str = ""
    topic: str = ""
    symbol_filter: Optional[str] = None  # Valore da matchare (es. ISIN)
    symbol_field: str = "instrument.isin"  # Path nel messaggio per il match
    store: str = "market"
    fields_mapping: Dict[str, str] = dataclass_field(default_factory=dict)
    
    def __post_init__(self):
        self.source = "kafka"
        if not self.topic:
            raise ValueError("topic is required for Kafka subscriptions")
        if not self.id:
            self.id = f"kafka_{self.topic.replace('.', '_')}"
            if self.symbol_filter:
                self.id += f"_{self.symbol_filter}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Kafka-specific fields"""
        base = super().to_dict()
        base.update({
            "id": self.id,
            "topic": self.topic,
            "symbol_filter": self.symbol_filter,
            "symbol_field": self.symbol_field,
            "store": self.store,
            "fields_mapping": self.fields_mapping
        })
        return base
    
    def matches(self, message: Dict[str, Any]) -> bool:
        """
        Verifica se il messaggio matcha il filtro della subscription.
        
        Args:
            message: Messaggio Kafka deserializzato
            
        Returns:
            True se il messaggio passa il filtro
        """
        if not self.symbol_filter:
            return True  # Nessun filtro, accetta tutto
        
        # Estrai il valore dal path (es. "instrument.isin")
        value = self._get_nested_value(message, self.symbol_field)
        return value == self.symbol_filter
    
    def extract_fields(self, message: Dict[str, Any]) -> Dict[str, float]:
        """
        Estrae i campi dal messaggio usando fields_mapping.
        
        Supporta nested paths come "bidBestLevel.price".
        
        Args:
            message: Messaggio Kafka deserializzato
            
        Returns:
            Dict con i campi estratti (target_field -> value)
        """
        result = {}
        
        for target_field, source_path in self.fields_mapping.items():
            value = self._get_nested_value(message, source_path)
            if value is not None:
                try:
                    result[target_field] = float(value)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values
        
        return result
    
    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
        """
        Estrae un valore da un dict nested usando un path dot-separated.
        
        Args:
            data: Dict sorgente
            path: Path tipo "bidBestLevel.price"
            
        Returns:
            Valore trovato o None
        """
        value = data
        for key in path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


@dataclass
class SubscriptionGroup:
    """
    Group of related subscriptions.

    Useful for managing subscriptions by category (e.g., equities, currencies, indices).
    """
    name: str
    subscriptions: Dict[str, LiveSubscription] = dataclass_field(default_factory=dict)
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def add_subscription(self, sub: LiveSubscription):
        """Add subscription to group"""
        self.subscriptions[sub.id] = sub

    def remove_subscription(self, ticker: str) -> Optional[LiveSubscription]:
        """Remove and return subscription"""
        return self.subscriptions.pop(ticker, None)

    def get_active_subscriptions(self) -> List[LiveSubscription]:
        """Get all active subscriptions"""
        return [s for s in self.subscriptions.values() if s.status == "active"]

    def get_failed_subscriptions(self) -> List[LiveSubscription]:
        """Get all failed subscriptions"""
        return [s for s in self.subscriptions.values() if s.status == "failed"]

    def get_stale_subscriptions(self, timeout_seconds: int = 60) -> List[LiveSubscription]:
        """Get subscriptions that haven't received data recently"""
        return [s for s in self.subscriptions.values() if s.is_stale(timeout_seconds)]

    def get_bloomberg_subscriptions(self) -> List[BloombergSubscription]:
        """Get only Bloomberg subscriptions from this group"""
        return [s for s in self.subscriptions.values()
                if isinstance(s, BloombergSubscription)]

    def get_redis_subscriptions(self) -> List[RedisSubscription]:
        """Get only RedisPublisher subscriptions from this group"""
        return [s for s in self.subscriptions.values()
                if isinstance(s, RedisSubscription)]


class LiveSubscriptionManager:
    """
    Advanced subscription manager using LiveSubscription dataclass.

    Features:
    - Track subscription lifecycle (pending -> active -> failed/closed)
    - Detect stale subscriptions
    - Group subscriptions by category
    - Rich metadata support
    """

    def __init__(self):
        # Live subscriptions by source (bloomberg, redis, kafka + data stores)
        self._live_subscriptions: Dict[str, Dict[str, LiveSubscription]] = {
            source: {} for source in [v.lower() for v in ["bloomberg", "redis", "kafka"]]
        }

        # Subscription groups (e.g., "equities", "currencies", "indices")
        self._subscription_groups: Dict[str, SubscriptionGroup] = {}

        # Legacy support
        self._subscription_status: Dict[str, Dict[str, Any]] = {}
        self._instrument_status: Dict[str, str] = {}
        self._logger = logging.getLogger(__name__)

        self._pending_subscriptions: Dict[str, Dict[str, LiveSubscription]] = {}
        self._to_unsubscribe: Dict[str, Dict[str, LiveSubscription]] = {}

    # SOSTITUISCI il metodo add_subscription esistente con questo:

    def add_subscription(self, subscription: LiveSubscription, group: Optional[str] = None):
        """
        Add a live subscription (starts as PENDING).

        Args:
            subscription: LiveSubscription instance
            group: Optional group name to add subscription to
        """
        source = subscription.source.lower()
        if source not in self._pending_subscriptions:
            self._logger.warning(f"Unknown source {source}, adding anyway")
            self._pending_subscriptions[source] = {}

        # Set status to pending
        subscription.status = "pending"

        # Add to PENDING first (NON a _live_subscriptions!)
        self._pending_subscriptions[source][subscription.id] = subscription

        # Add to group if specified
        if group:
            if group not in self._subscription_groups:
                self._subscription_groups[group] = SubscriptionGroup(group)
            self._subscription_groups[group].add_subscription(subscription)

        self._logger.info(f"Added subscription {subscription.id} to pending queue")

    def activate_subscription(self, id: str, source: str):
        """Move subscription from pending to active"""
        source = source.lower()

        # Remove from pending
        sub = self._pending_subscriptions.get(source, {}).pop(id, None)

        if sub:
            sub.mark_active()

            # Add to active subscriptions
            if source not in self._live_subscriptions:
                self._live_subscriptions[source] = {}
            self._live_subscriptions[source][id] = sub

            self._logger.info(f"Subscription {id} activated")
            return True

        return False

    def fail_subscription(self, id: str, source: str, error: str):
        """Move subscription to failed (from pending or active)"""
        source = source.lower()
        # Try pending first
        sub = self._pending_subscriptions.get(source, {}).pop(id, None)

        # Try active if not in pending
        if not sub:
            sub = self._live_subscriptions.get(source, {}).get(id)

        if sub:
            sub.mark_failed(error)
            
            # Ensure failed subscription is tracked in _live_subscriptions
            if source not in self._live_subscriptions:
                self._live_subscriptions[source] = {}
            self._live_subscriptions[source][id] = sub
            
            self._logger.warning(f"Subscription {id} failed: {error}")
            return True

        return False

    def mark_for_unsubscribe(self, id: str, source: str):
        """Mark subscription for removal"""
        source = source.lower()

        # Try to remove from active first
        sub = self._live_subscriptions.get(source, {}).pop(id, None)

        # If not in active, try pending
        if not sub:
            sub = self._pending_subscriptions.get(source, {}).pop(id, None)

        if sub:
            sub.mark_closed()

            # Add to unsubscribe queue
            if source not in self._to_unsubscribe:
                self._to_unsubscribe[source] = {}
            self._to_unsubscribe[source][id] = sub

            self._logger.info(f"Subscription {id} marked for unsubscribe")
            return True
        return False

    def get_pending_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """Get all pending subscriptions"""
        if source:
            return self._pending_subscriptions.get(source.lower(), {}).copy()

        # Merge all sources
        all_pending = {}
        for source_subs in self._pending_subscriptions.values():
            all_pending.update(source_subs)
        return all_pending

    def get_to_unsubscribe(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """Get all subscriptions marked for unsubscribe"""
        if source:
            return self._to_unsubscribe.get(source.lower(), {}).copy()

        # Merge all sources
        all_to_unsub = {}
        for source_subs in self._to_unsubscribe.values():
            all_to_unsub.update(source_subs)
        return all_to_unsub

    def clear_unsubscribed(self, id: str, source: str):
        """Remove from unsubscribe queue after processing"""
        source = source.lower()
        removed = self._to_unsubscribe.get(source, {}).pop(id, None)
        if removed:
            self._logger.info(f"Cleared unsubscribed {id}")
            return True
        return False

        # SOSTITUISCI il metodo get_subscription_health esistente con questo:

    def get_subscription_health(self) -> Dict[str, Any]:
        """
        Get overall subscription health metrics.

        Returns:
            Dict with counts of active, failed, stale subscriptions
        """
        all_subs = self.get_all_live_subscriptions()
        all_pending = self.get_pending_subscriptions()
        all_to_unsub = self.get_to_unsubscribe()

        return {
            "total": len(all_subs) + len(all_pending),
            "active": len([s for s in all_subs.values() if s.status == "active"]),
            "pending": len(all_pending),
            "failed": len([s for s in all_subs.values() if s.status == "failed"]),
            "closed": len([s for s in all_subs.values() if s.status == "closed"]),
            "to_unsubscribe": len(all_to_unsub),
            "stale": len([s for s in all_subs.values() if s.is_stale()]),
            "by_source": {
                source: {
                    "active": len(self._live_subscriptions.get(source, {})),
                    "pending": len(self._pending_subscriptions.get(source, {})),
                    "to_unsubscribe": len(self._to_unsubscribe.get(source, {}))
                }
                for source in DataStore.__members__
            }
        }

    # ========================================================================
    # NEW: LiveSubscription API
    # ========================================================================

    def remove_subscription(self, ticker: str, source: str) -> Optional[LiveSubscription]:
        """Remove and return subscription"""
        source = source.lower()
        if source in self._live_subscriptions:
            return self._live_subscriptions[source].pop(ticker, None)
        return None

    def get_subscription(self, ticker: str, source: str) -> Optional[LiveSubscription]:
        """Get subscription by ticker and source"""
        source = source.lower()
        return self._live_subscriptions.get(source, {}).get(ticker)

    def get_subscriptions_by_source(self, source: str) -> Dict[str, LiveSubscription]:
        """Get all subscriptions for a source"""
        return self._live_subscriptions.get(source.lower(), {}).copy()

    def get_all_live_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        """
        Get all live subscriptions.

        Args:
            source: Filter by source (None = all sources)

        Returns:
            Dict of ticker -> LiveSubscription
        """
        if source:
            return self._live_subscriptions.get(source.lower(), {}).copy()

        # Merge all sources
        all_subs = {}
        for source_subs in self._live_subscriptions.values():
            all_subs.update(source_subs)
        return all_subs

    def get_active_subscriptions(self, source: Optional[str] = None) -> List[LiveSubscription]:
        """Get all active subscriptions"""
        subs = self.get_all_live_subscriptions(source)
        return [s for s in subs.values() if s.status == "active"]

    def get_stale_subscriptions(self, timeout_seconds: int = 60,
                                source: Optional[str] = None) -> List[LiveSubscription]:
        """Get subscriptions that haven't received data recently"""
        subs = self.get_all_live_subscriptions(source)
        return [s for s in subs.values() if s.is_stale(timeout_seconds)]

    def mark_update_received(self, ticker: str, source: str):
        """Mark that data was received for a subscription"""
        sub = self.get_subscription(ticker, source)
        if sub:
            sub.update_received()

    def mark_subscription_failed(self, ticker: str, source: str, error: str):
        """Mark subscription as failed"""
        sub = self.get_subscription(ticker, source)
        if sub:
            sub.mark_failed(error)

    # ========================================================================
    # Groups API
    # ========================================================================

    def create_group(self, name: str, metadata: Optional[Dict] = None) -> SubscriptionGroup:
        """Create a subscription group"""
        if name not in self._subscription_groups:
            self._subscription_groups[name] = SubscriptionGroup(name, metadata=metadata or {})
        return self._subscription_groups[name]

    def get_group(self, name: str) -> Optional[SubscriptionGroup]:
        """Get subscription group by name"""
        return self._subscription_groups.get(name)

    def get_all_groups(self) -> Dict[str, SubscriptionGroup]:
        """Get all subscription groups"""
        return self._subscription_groups.copy()

    # ========================================================================
    # LEGACY API (backward compatibility)
    # ========================================================================

    def set_bloomberg_subscriptions(self, subscriptions: Union[Dict[str, str], pd.Series],
                                    fields: Optional[List[str]] = None,
                                    params: Optional[Dict[str, Any]] = None):
        """
        Set Bloomberg subscriptions (legacy API).

        Args:
            subscriptions: Dict of ticker -> subscription string
            fields: Optional list of fields to subscribe to
            params: Optional Bloomberg parameters
        """
        if isinstance(subscriptions, pd.Series):
            subscriptions = subscriptions.to_dict()

        fields = fields or ["BID", "ASK"]
        params = params or {}

        subscription_set = set()
        for id, subscription_str in subscriptions.items():
            if subscription_str in subscription_set:
                self._logger.error(f"Duplicate subscription {subscription_str} for {id}")
            else:
                # Create BloombergSubscription
                bloomberg_sub = BloombergSubscription(
                    id=id,
                    subscription_string=subscription_str,
                    fields=fields.copy(),
                    params=params.copy()
                )
                self.add_subscription(bloomberg_sub)
                subscription_set.add(subscription_str)

    def set_redis_subscriptions(self, subscriptions: List[str],
                                fields: Optional[List[str]] = None):
        """
        Set RedisPublisher channel subscriptions (legacy API).

        Args:
            subscriptions: List of channel names or patterns
            fields: Optional list of fields
        """
        for channel in subscriptions:
            redis_sub = RedisSubscription(
                id=f"redis_{channel.replace(':', '_')}",
                channel=channel,
                subscription=channel,
                fields=fields
            )
            self.add_subscription(redis_sub)

    def get_bloomberg_subscriptions(self) -> Dict[str, str]:
        """Get Bloomberg subscriptions (legacy API - returns ticker -> subscription_string)"""
        subs = self._live_subscriptions.get("bloomberg", {})
        return {
            ticker: sub.subscription_string
            for ticker, sub in subs.items()
            if isinstance(sub, BloombergSubscription)
        }

    def get_redis_subscriptions(self) -> List[str]:
        """Get RedisPublisher subscriptions (legacy API - returns list of channels)"""
        subs = self._live_subscriptions.get("redis", {})
        return [
            sub.channel
            for sub in subs.values()
            if isinstance(sub, RedisSubscription)
        ]

    def get_all_subscriptions(self) -> Dict[str, Any]:
        """Get all subscriptions (legacy API)"""
        return {
            "bloomberg": self.get_bloomberg_subscriptions(),
            "redis": self.get_redis_subscriptions()
        }

    def update_subscription_status(self, isin: str, status: Dict[str, Any]):
        """Update subscription status (legacy API)"""
        if isin not in self._subscription_status:
            self._subscription_status[isin] = {}
        self._subscription_status[isin].update(status)

        # Update LiveSubscription if exists
        for source in self._live_subscriptions.values():
            if isin in source:
                if "is_delayed_stream" in status:
                    source[isin].is_delayed = (status["is_delayed_stream"] == "true")

    def update_instrument_status(self, isin: str, status: str):
        """Update instrument status (legacy API)"""
        self._instrument_status[isin] = status

    def get_subscription_status(self, isin: Optional[str] = None) -> Union[Dict, Any]:
        """Get subscription status (legacy API)"""
        if isin is None:
            return self._subscription_status.copy()
        return self._subscription_status.get(isin, {})

    def get_instrument_status(self, isin: Optional[str] = None) -> Union[Dict[str, str], str]:
        """Get instrument status (legacy API)"""
        if isin is None:
            return self._instrument_status.copy()
        return self._instrument_status.get(isin, "UNKNOWN")

    def get_delayed_status(self) -> pd.Series:
        """Get delayed stream status (legacy API)"""
        return pd.Series({
            isin: data.get("is_delayed_stream", None) == "true"
            for isin, data in self._subscription_status.items()
        })

    # ========================================================================
    # Monitoring & Health
    # ========================================================================

    def get_failed_subscriptions_report(self) -> List[Dict[str, Any]]:
        """Get detailed report of failed subscriptions"""
        all_subs = self.get_all_live_subscriptions()
        failed = [s for s in all_subs.values() if s.status == "failed"]

        return [
            {
                "id": s.id,
                "source": s.source,
                "error_count": s.error_count,
                "last_error": s.metadata.get("last_error"),
                "last_error_time": s.metadata.get("last_error_time")
            }
            for s in failed
        ]