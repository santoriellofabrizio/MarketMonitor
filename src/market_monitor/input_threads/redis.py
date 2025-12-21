"""
redis con routing ibrido finale.

REGOLE ROUTING:
1. Store detection:
   - Subscription.store (priorità 1)
   - Primo segmento channel se in {market, state, events, blob} (priorità 2)
   - Default: "market" (priorità 3)

2. MarketStore (DataFrame):
   - Format: {ticker: fields_dict} o {ticker: scalar}
   - Supporta 1 livello nesting: {ticker: {field: value}}
   - Path dal channel non usato per market

3. StateStore (Nested Dict):
   - Path dal channel → nested structure
   - Deep update automatico

4. EventStore (Deque):
   - Append data
   - Event type dal channel o state_namespace

5. BlobStore (Key-Value):
   - Key dall'ultimo segmento channel
"""

import json
import logging
import redis
import threading
from typing import Optional, Any, Dict, List

from market_monitor.live_data_hub.real_time_data_hub import RTData, RedisSubscription

logger = logging.getLogger(__name__)


# ============================================================================
# Parser Universale
# ============================================================================

def parse_redis_message(json_str: str) -> tuple[Any, Dict[str, Any]]:
    """Parse messaggio RedisPublisher in formato flat o wrapped."""
    try:
        msg = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Errore parsing JSON: {e}")
        raise

    # Formato flat (nuovo)
    if "__metadata__" in msg:
        metadata = msg.pop("__metadata__")

        if "records" in msg:
            data = msg["records"]
        elif "__value__" in msg:
            data = msg["__value__"]
        else:
            data = msg

        return data, metadata

    # Formato wrapped (legacy)
    elif "data" in msg:
        data = msg["data"]
        metadata = {k: v for k, v in msg.items() if k != "data"}
        return data, metadata

    # Formato semplice (no metadata)
    else:
        logger.debug("Messaggio senza metadata rilevato")
        return msg, {}


# ============================================================================
# Utility: Nested Dict Construction
# ============================================================================

def build_nested_dict(path: List[str], value: Any) -> Dict:
    """
    Costruisce nested dict da path e value.

    Esempio:
        path = ["portfolio", "cash"]
        value = 1000000
        → {"portfolio": {"cash": 1000000}}
    """
    if not path:
        return value if isinstance(value, dict) else {"__value__": value}

    result = {}
    current = result

    for key in path[:-1]:
        current[key] = {}
        current = current[key]

    # Ultimo livello
    current[path[-1]] = value

    return result


def deep_update(target: Dict, source: Dict) -> None:
    """Deep merge di source in target (modifica target in-place)."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = value


# ============================================================================
# redis FINALE
# ============================================================================

class RedisStreamingThread(threading.Thread):

    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, channels=None):
        """
        Thread per RedisPublisher pub/sub con routing ibrido.

        REGOLE:
        - MarketStore: ticker-based (flat DataFrame)
        - Altri store: self-describing (nested dict)
        """
        super().__init__(name="redis")
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)
        self.pubsub = self.redis_client.pubsub()
        self.channels: set = set(channels) if channels else set()
        self.stop_event = threading.Event()
        self.real_time_data: Optional[RTData] = None
        self.subscription_service = None
        self.subscriptions: Dict[str, RedisSubscription] = {}  # pattern -> subscription

        # Store keywords
        self.store_keywords = {"market", "state", "events", "blob"}

    def set_real_time_data(self, real_time_data: RTData) -> None:
        """
        Configura RTData e carica le sottoscrizioni RedisPublisher.

        VALIDAZIONE: Ogni subscription pattern deve mappare a UN SOLO store.
        """
        self.real_time_data = real_time_data
        self.subscription_service = real_time_data.get_subscription_manager()

        # Ottieni pending + active subscriptions
        pending = self.subscription_service.get_pending_subscriptions("redis") or {}
        active = self.subscription_service.get_redis_subscription() or {}
        all_subs = {**pending, **active}

        # Crea mapping pattern -> store con validazione
        pattern_store_map = {}

        for sub_id, sub in all_subs.items():
            if isinstance(sub, RedisSubscription):
                pattern = sub.subscription

                # Determina store effettivo (per validation)
                store = self._determine_store(sub.channel, sub)

                # VALIDAZIONE: Check conflitto store
                if pattern in pattern_store_map:
                    existing_store = pattern_store_map[pattern]
                    if existing_store != store:
                        existing_sub = self.subscriptions[pattern]
                        raise ValueError(
                            f"❌ CONFLITTO: pattern '{pattern}' mappato a store multipli:\n"
                            f"   - '{existing_store}' (subscription: {existing_sub.id})\n"
                            f"   - '{store}' (subscription: {sub_id})\n"
                            f"   → Ogni pattern può avere UN SOLO store!"
                        )
                    logger.debug(f"Subscription duplicata ignorata: {sub_id} su {pattern}")
                    continue

                # Registra mapping univoco
                pattern_store_map[pattern] = store
                self.subscriptions[pattern] = sub
                self.channels.add(sub.channel)

        logger.info(
            f"redis configured: {len(self.subscriptions)} patterns\n"
            f"  Store mapping: {pattern_store_map}"
        )

    def run(self):
        """Loop principale: ascolta messaggi e routa."""
        try:
            if not self.channels:
                logger.warning("Nessun pattern RedisPublisher specificato")
                return

            # Subscribe con pattern support
            self.pubsub.psubscribe(*self.channels)
            logger.info(f"Sottoscritto ai pattern RedisPublisher: {self.channels}")

            for message in self.pubsub.listen():
                if self.stop_event.is_set():
                    break

                if message['type'] in ('message', 'pmessage'):
                    self._handle_message(message)

        except Exception as e:
            logger.error(f"Errore in redis: {e}", exc_info=True)

    def _handle_message(self, message):
        """Gestisce messaggio con routing ibrido."""
        try:
            channel = message['channel'].decode('utf-8')
            raw_data = message['data'].decode('utf-8')

            # Parser universale
            data, metadata = parse_redis_message(raw_data)

            # Trova subscription
            subscription = self._find_subscription_for_channel(channel)
            if not subscription:
                logger.warning(f"No subscription found for channel '{channel}'")
                return

            # Determina store e path
            store = self._determine_store(channel, subscription)
            path = self._extract_path(channel, store)

            # Route data
            self._route_data(subscription, store, path, channel, data, metadata)

            # Mark received
            if self.subscription_service:
                self.subscription_service.mark_subscription_received(subscription.id, "redis")

        except Exception as e:
            logger.error(f"Errore handling RedisPublisher message: {e}", exc_info=True)

    def _determine_store(self, channel: str, subscription: RedisSubscription) -> str:
        """
        Determina store target.

        Priorità:
        1. subscription.store (se specificato)
        2. Primo segmento del channel (se è store keyword)
        3. Default: "market"
        """
        # Priorità 1: Subscription esplicita
        if subscription.store:
            return subscription.store

        # Priorità 2: Self-describing
        first_segment = channel.split(':')[0] if ':' in channel else channel
        if first_segment in self.store_keywords:
            return first_segment

        # Priorità 3: Default
        logger.debug(
            f"Cannot determine store from channel '{channel}'. "
            f"First segment '{first_segment}' not in {self.store_keywords}. "
            f"Defaulting to 'market'."
        )
        return "market"

    def _extract_path(self, channel: str, store: str) -> List[str]:
        """
        Estrai path dal channel.

        Se primo segmento è store keyword, skippa.
        Altrimenti, usa tutto il channel.
        """
        segments = channel.split(':')

        # Se primo segmento è store keyword, skippa
        if segments[0] in self.store_keywords:
            return segments[1:]  # Resto del path

        # Altrimenti usa tutto
        return segments

    def _route_data(self, sub: RedisSubscription, store: str, path: List[str],
                   channel: str, data: Any, metadata: Dict):
        """Route data basato su store."""
        msg_type = metadata.get('type', 'UNKNOWN')
        logger.debug(f"Routing {msg_type} to '{store}' store (path={path})")

        if store == "market":
            self._route_to_market(sub, path, data)

        elif store == "state":
            self._route_to_state(sub, path, data)

        elif store == "events":
            self._route_to_events(sub, channel, data)

        elif store == "blob":
            self._route_to_blob(sub, path, data)

        else:
            logger.warning(f"Unknown store type '{store}' for {sub.id}")

    def _route_to_market(self, sub: RedisSubscription, path: List[str], data: Any):
        """
        Route to MarketStore (ticker-based, DataFrame).

        FORMATO ATTESO:
        1. Dict of dicts: {ticker: {field: value, ...}}
        2. Dict of scalars: {ticker: value} → usa path come field name
        3. Supporta 1-level nesting: {ticker: {field_name: value}}

        ESEMPI:
        data = {"ISIN123": {"BID": 100, "ASK": 101}}
        → rtdata.update("ISIN123", {"BID": 100, "ASK": 101})

        data = {"ISIN123": 100.5}, path=["nav"]
        → rtdata.update("ISIN123", {"nav": 100.5})

        data = {"EUR": {"EONIA": 100.5}}, path=["EUR", "EONIA"]
        → rtdata.update("EUR", {"EONIA": 100.5})  # 1-level nesting
        """
        if not isinstance(data, dict):
            logger.warning(f"Market data must be dict, got {type(data)}")
            return

        for ticker, value in data.items():
            # Skip metadata fields
            if ticker.startswith('_'):
                continue

            # Case 1: value è dict (multi-field o nested)
            if isinstance(value, dict):
                # Check se nested (sub-values sono numerici o dict)
                first_val = next(iter(value.values())) if value else None

                if isinstance(first_val, dict):
                    # Nested 1-level: {"EUR": {"EONIA": {"BID": 1.5}}}
                    # → ticker="EUR", fields={"EONIA": {...}}
                    fields = value
                else:
                    # Flat multi-field: {"ISIN123": {"BID": 100, "ASK": 101}}
                    # Converti a float
                    fields = {}
                    for k, v in value.items():
                        if not k.startswith('_'):
                            try:
                                fields[k] = float(v)
                            except (ValueError, TypeError):
                                logger.debug(f"Could not convert {k}={v} to float, keeping as-is")
                                fields[k] = v

            # Case 2: value è scalar
            else:
                # Usa path come field name (ultimo segmento)
                field_name = path[-1] if path else "value"
                try:
                    fields = {field_name: float(value)}
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {ticker}={value} to float")
                    continue

            # Update RTData
            self.real_time_data.update(ticker, fields, store="market")
            logger.debug(f"MarketStore updated: {ticker} = {fields}")

    def _route_to_state(self, sub: RedisSubscription, path: List[str], data: Any):
        """
        Route to StateStore (self-describing nested dict).

        FORMATO:
          path = ["portfolio", "cash"]
          data = 1000000
          → StateStore: {"portfolio": {"cash": 1000000}}
        """
        if not path:
            logger.warning("State data requires path segments from channel")
            return

        # Costruisci nested dict
        nested = build_nested_dict(path, data)

        # Deep update in StateStore
        with self.real_time_data.locker:
            deep_update(self.real_time_data._state_store._data, nested)

        logger.debug(f"StateStore updated: {path} = {data}")

    def _route_to_events(self, sub: RedisSubscription, channel: str, data: Any):
        """
        Route to EventStore (append).

        Usa channel o state_namespace come event_type.
        """
        event_type = sub.state_namespace or channel

        self.real_time_data._event_store.append(event_type, data)
        logger.debug(f"EventStore appended: {event_type}")

    def _route_to_blob(self, sub: RedisSubscription, path: List[str], data: Any):
        """
        Route to BlobStore.

        Usa path (joined) come key.
        """
        key = ":".join(path) if path else "default"

        self.real_time_data._blob_store.store(key, data)
        logger.debug(f"BlobStore stored: {key}")

    def _find_subscription_for_channel(self, channel: str) -> Optional[RedisSubscription]:
        """
        Trova subscription per canale (supporta pattern con wildcard).

        GARANTITO: ritorna al massimo 1 subscription (1 pattern = 1 store).
        """
        # 1. Exact match
        if channel in self.subscriptions:
            logger.debug(f"Exact match found for channel '{channel}'")
            return self.subscriptions[channel]

        # 2. Pattern match (solo per pattern con wildcard)
        for pattern, sub in self.subscriptions.items():
            if '*' in pattern and self._matches_pattern(channel, pattern):
                logger.debug(f"Pattern match: '{channel}' matched by '{pattern}'")
                return sub

        # Nessun match
        logger.debug(
            f"No subscription found for channel '{channel}'. "
            f"Available patterns: {list(self.subscriptions.keys())}"
        )
        return None

    @staticmethod
    def _matches_pattern(channel: str, pattern: str) -> bool:
        """
        Simple pattern matching (supports '*' wildcard).

        Esempi:
          - "market:*" matches "market:prices", "market:nav"
          - "state:portfolio:*" matches "state:portfolio:cash"
        """
        if '*' not in pattern:
            return channel == pattern

        # Convert pattern to regex
        import re
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(f"^{regex_pattern}$", channel))

    def stop(self):
        """Ferma il thread."""
        self.stop_event.set()
        self.pubsub.close()
        logger.info("redis terminato")


# ============================================================================
# Esempio di utilizzo
# ============================================================================

if __name__ == '__main__':
    from market_monitor.live_data_hub.real_time_data_hub import RTData
    from threading import Lock

    # Setup RTData
    lock = Lock()
    rtdata = RTData(lock, fields=["BID", "ASK", "nav"])

    # ========================================================================
    # MARKET: ticker-based
    # ========================================================================

    # Esempio 1: Subscription-based store
    rtdata.subscribe_redis(
        id="market_prices",
        channel="prices",  # Non self-describing
        subscription="prices:*",
        store="market"  # Esplicito
    )

    # Esempio 2: Self-describing
    rtdata.subscribe_redis(
        id="market_nav",
        channel="market:nav",  # Self-describing
        subscription="market:*",
        store=None  # Auto-detect: "market"
    )

    # ========================================================================
    # STATE: self-describing nested dict
    # ========================================================================
    rtdata.subscribe_redis(
        id="portfolio_state",
        channel="state:portfolio",
        subscription="state:*",
        store=None  # Auto-detect: "state"
    )

    # ========================================================================
    # EVENTS: append-only
    # ========================================================================
    rtdata.subscribe_redis(
        id="trade_events",
        channel="events:trades",
        subscription="events:*",
        store=None  # Auto-detect: "events"
    )

    # Start thread
    redis_thread = RedisStreamingThread(redis_host='localhost', redis_port=6379)
    redis_thread.set_real_time_data(rtdata)
    redis_thread.start()

    # Simula messaggi
    import time
    r = redis.StrictRedis()

    # 1. Market - multi-field (flat)
    r.publish("prices", json.dumps({
        "ISIN123": {"BID": 100.0, "ASK": 101.0},
        "ISIN456": {"BID": 200.0, "ASK": 201.0},
        "__metadata__": {"type": "DATA"}
    }))

    # 2. Market - scalar
    r.publish("market:nav", json.dumps({
        "ISIN123": 100.5,
        "ISIN456": 200.3,
        "__metadata__": {"type": "DATA"}
    }))

    # 3. Market - 1-level nesting
    r.publish("market:EUR:EONIA", json.dumps({
        "EUR": {"EONIA": 100.5},
        "__metadata__": {"type": "DATA"}
    }))

    # 4. State - nested dict
    r.publish("state:portfolio:cash", json.dumps({
        "__value__": 1000000,
        "__metadata__": {"type": "STATE"}
    }))

    time.sleep(2)
    redis_thread.stop()
    redis_thread.join()

    # Verifica dati
    print("\n=== MarketStore (DataFrame) ===")
    print(rtdata.get_data_field(index_data="market"))

    print("\n=== StateStore (Nested Dict) ===")
    print(rtdata._state_store.get_all())
