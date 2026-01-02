import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from market_monitor.live_data_hub.live_subscription import (
    BloombergSubscription,
    LiveSubscription,
    LiveSubscriptionManager,
    RedisSubscription,
    SubscriptionGroup,
)

# Costanti riutilizzate dalla vecchia gestione in RTData
SECURITY_TO_IGNORE = ["GBp", "OTHEREQUIEUR", "ILs"]
EUR_SYNONYM = ["EUREUR", "EUREUR CURNCY"]

logger = logging.getLogger(__name__)

class SubscriptionService:
    """
    Servizio che incapsula la gestione delle sottoscrizioni.

    - Centralizza LiveSubscriptionManager
    - Valida le liste di strumenti
    - Espone API per health-check e lifecycle
    """

    def __init__(self, manager: Optional[LiveSubscriptionManager] = None):
        self._manager = manager or LiveSubscriptionManager()
        self._logger = logging.getLogger(__name__)
        self._allowed_securities: set[str] = set()

    # ------------------------------------------------------------------
    # Securities validation
    # ------------------------------------------------------------------
    def normalize_securities(self, securities: Union[str, List[str]]) -> List[str]:
        """Normalizza la lista strumenti (filtra vuoti, sinonimi e duplicati)."""
        if isinstance(securities, str):
            securities = [securities]

        cleaned: List[str] = []
        seen: set[str] = set()

        for sec in securities:
            if not sec:
                continue

            if sec in EUR_SYNONYM:
                self._logger.warning("Use 'EUR' instead of %s", sec)
                continue

            if sec in SECURITY_TO_IGNORE:
                continue

            if sec not in seen:
                cleaned.append(sec)
                seen.add(sec)

        return cleaned

    def set_allowed_securities(self, securities: Union[str, List[str]]) -> List[str]:
        normalized = self.normalize_securities(securities)
        self._allowed_securities = set(normalized)
        return normalized

    def _validate_subscription_target(self, id: Optional[str]) -> bool:
        if not id:
            self._logger.error("Subscription id cannot be empty")
            return False

        if self._allowed_securities and id not in self._allowed_securities:
            self._logger.warning("Subscription %s not in allowed securities", id)
            return False

        return True

    # ------------------------------------------------------------------
    # Subscription lifecycle API
    # ------------------------------------------------------------------
    def subscribe_bloomberg(
        self,
        id: str,
        subscription_string: Optional[str] = None,
        fields: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
    ) -> BloombergSubscription:
        if not self._validate_subscription_target(id):
            logger.warning("Subscription %s invalid", id)

        bloomberg_sub = BloombergSubscription(
            id=id,
            subscription_string=subscription_string or id,
            fields=fields or ["BID", "ASK"],
            params=params or {},
        )
        self._manager.add_subscription(bloomberg_sub, group=group)
        return bloomberg_sub

    def subscribe_redis(
        self,
        channel: str,
        subscription: Optional[str] = None,
        store: Optional[str] = None,
        event_type: Optional[str] = None,
        fields: Optional[List[str]] = None,
        id: Optional[str] = None,
        group: Optional[str] = None,
    ) -> RedisSubscription:
        redis_sub = RedisSubscription(
            id=id or f"{channel.replace(':', '_')}",
            channel=channel,
            subscription=subscription or channel,
            store=store,
            event_type=event_type,
            fields=fields,
        )
        self._manager.add_subscription(redis_sub, group=group)
        return redis_sub

    def unsubscribe(self, id: str, source: str) -> bool:
        return self._manager.mark_for_unsubscribe(id, source)

    def mark_subscription_received(self, id: str, source: str = "bloomberg"):
        activated = self._manager.activate_subscription(id, source)
        if not activated:
            self._manager.mark_update_received(id, source)

    def mark_subscription_failed(self, id: str, source: str, error: Optional[str] = None):
        self._manager.fail_subscription(id, source, error or "Unknown error")

    def clear_unsubscribed(self, id: str, source: str):
        self._manager.clear_unsubscribed(id, source)

    # ------------------------------------------------------------------
    # Accessors & queries
    # ------------------------------------------------------------------
    def get_bloomberg_subscription(
        self, ticker: Optional[str] = None
    ) -> Optional[Dict[str, BloombergSubscription]]:
        if ticker:
            sub = self._manager.get_subscription(ticker, "bloomberg")
            if isinstance(sub, BloombergSubscription):
                return {ticker: sub}
            return None
        return self._manager.get_subscriptions_by_source("bloomberg")

    def get_redis_subscription(
        self, ticker: Optional[str] = None
    ) -> Optional[Dict[str, RedisSubscription]]:
        if ticker:
            sub = self._manager.get_subscription(ticker, "redis")
            if isinstance(sub, RedisSubscription):
                return {ticker: sub}
            return None
        return self._manager.get_subscriptions_by_source("redis")

    def get_pending_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        return self._manager.get_pending_subscriptions(source)

    def get_to_unsubscribe(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        return self._manager.get_to_unsubscribe(source)

    def get_all_subscriptions(self, source: Optional[str] = None) -> Dict[str, LiveSubscription]:
        return self._manager.get_all_live_subscriptions(source)

    def get_active_subscriptions(self, source: Optional[str] = None):
        return self._manager.get_active_subscriptions(source)

    def get_subscription(self, ticker: str, source: str) -> Optional[LiveSubscription]:
        return self._manager.get_subscription(ticker, source)

    def get_subscription_health(self) -> Dict[str, Any]:
        return self._manager.get_subscription_health()

    def get_failed_subscriptions(self) -> List[Dict[str, Any]]:
        return self._manager.get_failed_subscriptions_report()

    def get_subscription_group(self, name: str) -> Optional[SubscriptionGroup]:
        return self._manager.get_group(name)

    def create_subscription_group(self, name: str, metadata: Optional[Dict] = None) -> SubscriptionGroup:
        return self._manager.create_group(name, metadata)

    def get_delayed_status(self) -> pd.Series:
        return self._manager.get_delayed_status()

    # ------------------------------------------------------------------
    # Legacy dict-style helpers
    # ------------------------------------------------------------------
    @property
    def subscription_status(self) -> Dict[str, Dict[str, Any]]:
        return self._manager.get_subscription_status()

    @subscription_status.setter
    def subscription_status(self, value: Dict[str, Dict[str, Any]]):
        self._manager._subscription_status = value or {}

    @property
    def instrument_status(self) -> Dict[str, str]:
        return self._manager._instrument_status

    def update_instrument_status(self, isin: str, status: str):
        self._manager.update_instrument_status(isin, status)

    def set_bloomberg_subscriptions(
        self,
        subscriptions: Union[Dict[str, str], pd.Series],
        fields: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(subscriptions, pd.Series):
            subscriptions = subscriptions.to_dict()

        validated = {isin: sub for isin, sub in subscriptions.items() if self._validate_subscription_target(isin)}
        if not validated:
            return
        self._manager.set_bloomberg_subscriptions(validated, fields, params)

    def set_redis_subscriptions(self, subscriptions: List[str], fields: Optional[List[str]] = None):
        self._manager.set_redis_subscriptions(subscriptions, fields)

    def apply_subscription_dict(self, value: Dict[str, Any]):
        for source, subs in value.items():
            if source.lower() == "bloomberg":
                self.set_bloomberg_subscriptions(subs)
            elif source.lower() == "redis":
                self.set_redis_subscriptions(subs)
            else:
                self._logger.warning("Unknown subscription source: %s", source)

    def get_all_subscriptions_dict(self) -> Dict[str, Any]:
        return self._manager.get_all_subscriptions()

