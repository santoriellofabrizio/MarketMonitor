"""
Kafka Streaming Thread per MarketMonitor.

Si connette a un cluster Kafka e riceve dati di mercato in real-time.
I messaggi sono deserializzati con Avro via Schema Registry.

CONFIGURAZIONE:
- Le sottoscrizioni Kafka vengono gestite tramite RTData.get_subscription_manager().subscribe_kafka()
- Ogni subscription specifica: topic, symbol_filter (ISIN), store target, fields_mapping

ESEMPIO:
    from threading import Lock
    from market_monitor.live_data_hub.real_time_data_hub import RTData
    from market_monitor.input_threads import KafkaStreamingThread

    lock = Lock()
    rtdata = RTData(locker=lock, fields=["BID", "ASK"])

    # Subscribe a ETP via ISIN
    rtdata.get_subscription_manager().subscribe_kafka(
        id="IWDA",
        topic="COALESCENT_DUMA.ETFP.BookBest",
        symbol_filter="IE00B4L5Y983",
        symbol_field="instrument.isin",
        store="market",
        fields_mapping={
            "BID": "bidBestLevel.price",
            "ASK": "askBestLevel.price",
            "BID_SIZE": "bidBestLevel.quantity",
            "ASK_SIZE": "askBestLevel.quantity"
        }
    )

    kafka_thread = KafkaStreamingThread(rtdata)
    kafka_thread.start()
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from queue import Queue
from typing import Optional, Dict, Any, List, Set, Tuple, Deque

import pandas as pd

from market_monitor.live_data_hub.real_time_data_hub import RTData
from market_monitor.live_data_hub.live_subscription import KafkaSubscription
from market_monitor.live_data_hub.subscription_service import SubscriptionService
from market_monitor.input_threads.trade import TradeType

logger = logging.getLogger(__name__)

# Type alias for trade deduplication key: (isin, market, timestamp, price, quantity)
TradeKey = Tuple[str, str, int, float, float]


class _KafkaBaseThread(threading.Thread):
    """
    Base class for Kafka streaming threads.

    Handles the common Kafka setup, polling loop, subscription loading,
    and hot-reload logic.  Subclasses must implement:
        - _should_include_subscription(sub): which subscriptions this thread owns
        - _process_message(topic, value): what to do with a deserialized message

    Optional hooks:
        - _on_poll_cycle(): called on every poll iteration (before None-check)
        - _on_poll_idle(): called when poll returns None (default: check new subs)
    """

    DEFAULT_BOOTSTRAP_SERVERS = "aftstserver51.af.pre:9092,aftstserver52.af.pre:9092,aftstserver53.af.pre:9092"
    DEFAULT_SCHEMA_REGISTRY = "http://aftstserver51.af.pre:8081,http://aftstserver52.af.pre:8081,http://aftstserver53.af.pre:8081"

    def __init__(self,
                 subscription_service: SubscriptionService,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 **kwargs):
        super().__init__(daemon=True)

        self._subscription_service = subscription_service
        self.bootstrap_servers = bootstrap_servers or self.DEFAULT_BOOTSTRAP_SERVERS
        self.schema_registry_url = schema_registry_url or self.DEFAULT_SCHEMA_REGISTRY
        self.start_mode = start_mode
        self.consumer_group = consumer_group or str(uuid.uuid4())

        self.stop_event = threading.Event()
        self.running = False
        self.consumer = None
        self.avro_deserializer = None

        # topic -> [KafkaSubscription]
        self._subscriptions_by_topic: Dict[str, List[KafkaSubscription]] = {}
        self._topics: Set[str] = set()
        # topic -> symbol_filter -> KafkaSubscription  (O(1) fast path)
        self._isin_to_sub: defaultdict[str, Dict[str, KafkaSubscription]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _should_include_subscription(self, sub: KafkaSubscription) -> bool:
        """Return True if this thread should handle *sub*."""
        raise NotImplementedError

    def _process_message(self, topic: str, value: Dict[str, Any]):
        """Dispatch a fully-deserialized Kafka message."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _on_poll_cycle(self):
        """Called on every poll iteration. Override for per-cycle work."""

    def _on_poll_idle(self):
        """Called when poll returns None (no message)."""
        self._check_new_subscriptions()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def _load_subscriptions(self):
        """Load (or reload) subscriptions from the shared SubscriptionService."""
        pending = self._subscription_service.get_pending_subscriptions("kafka") or {}
        active = self._subscription_service.get_kafka_subscription() or {}
        all_subs = {**pending, **active}

        self._subscriptions_by_topic.clear()
        self._topics.clear()
        self._isin_to_sub.clear()

        count = 0
        for sub in all_subs.values():
            if isinstance(sub, KafkaSubscription) and self._should_include_subscription(sub):
                topic = sub.topic
                self._subscriptions_by_topic.setdefault(topic, []).append(sub)
                self._topics.add(topic)
                if sub.symbol_filter:
                    self._isin_to_sub[topic][sub.symbol_filter] = sub
                count += 1

        logger.info(
            f"{self.__class__.__name__}: loaded {count} subscriptions "
            f"for {len(self._topics)} topics: {self._topics}"
        )

    def _check_new_subscriptions(self):
        """Detect newly registered subscriptions and re-subscribe the consumer."""
        pending = self._subscription_service.get_pending_subscriptions("kafka") or {}
        new_topics = {
            sub.topic
            for sub in pending.values()
            if isinstance(sub, KafkaSubscription)
               and self._should_include_subscription(sub)
               and sub.topic not in self._topics
        }
        if new_topics:
            logger.info(f"{self.__class__.__name__}: new topics detected: {new_topics}")
            self._load_subscriptions()
            if self.consumer:
                self.consumer.subscribe(list(self._topics))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        try:
            from confluent_kafka import Consumer
            from confluent_kafka.schema_registry import SchemaRegistryClient
            from confluent_kafka.schema_registry.avro import AvroDeserializer
        except ImportError as e:
            logger.error(f"Kafka libraries not installed: {e}")
            logger.error("Install with: pip install confluent_kafka fastavro")
            return

        self._load_subscriptions()

        if not self._topics:
            logger.warning(f"{self.__class__.__name__}: no topics, thread idle")
            while not self.stop_event.wait(timeout=5):
                self._load_subscriptions()
                if self._topics:
                    break
            if not self._topics:
                logger.info(f"{self.__class__.__name__} stopping (no subscriptions)")
                return

        schema_registry_client = SchemaRegistryClient({"url": self.schema_registry_url})
        self.avro_deserializer = AvroDeserializer(schema_registry_client=schema_registry_client)

        consumer_conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.consumer_group,
            "auto.offset.reset": self.start_mode,
        }
        self.consumer = Consumer(consumer_conf)
        self.consumer.subscribe(list(self._topics))

        logger.info(f"{self.__class__.__name__} started - topics: {self._topics}")
        self.running = True

        deserializer = self.avro_deserializer
        stop_event = self.stop_event
        consumer = self.consumer

        try:
            while not stop_event.is_set():
                msg = consumer.poll(timeout=0.1)
                self._on_poll_cycle()

                if msg is None:
                    self._on_poll_idle()
                    continue

                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                try:
                    value = deserializer(msg.value())
                except Exception as e:
                    logger.debug(f"Failed to deserialize message: {e}")
                    continue

                self._process_message(msg.topic(), value)

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
            self.running = False
            logger.info(f"{self.__class__.__name__} stopped")

    def stop(self):
        self.stop_event.set()
        logger.info(f"{self.__class__.__name__} stop requested")


class KafkaStreamingThread(_KafkaBaseThread):
    """
    Thread per streaming dati da Kafka con deserializzazione Avro.

    Segue lo stesso pattern di BloombergStreamingThread e RedisStreamingThread.

    Features:
    - Deserializzazione Avro via Schema Registry
    - Filtraggio client-side per symbol (ISIN, ticker, etc.)
    - Supporto nested fields (es. "bidBestLevel.price")
    - Routing a store multipli (market, state, events, blob)
    - Hot-reload subscriptions

    Args:
        real_time_data: RTData instance per storage dati
        bootstrap_servers: Lista server Kafka (default: aftstserver51-53)
        schema_registry_url: URL Schema Registry per deserializzazione Avro
        start_mode: "latest" (nuovi messaggi) o "earliest" (dall'inizio)
        consumer_group: Group ID per il consumer (default: UUID random)
    """

    def __init__(self,
                 real_time_data: RTData,
                 subscription_service: SubscriptionService,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 **kwargs):
        super().__init__(subscription_service, bootstrap_servers, schema_registry_url,
                         start_mode, consumer_group, **kwargs)
        self.name = "kafka"
        self.real_time_data = real_time_data

    def _should_include_subscription(self, sub: KafkaSubscription) -> bool:
        return True

    def _process_message(self, topic: str, value: Dict[str, Any]):
        instrument = value.get('instrument')
        if instrument:
            topic_subs = self._isin_to_sub.get(topic, {})
            for v in instrument.values():
                if sub := topic_subs.get(v):
                    self._process_matched_message(sub, value, self.real_time_data)

    def _process_matched_message(self, sub: KafkaSubscription, value: Dict[str, Any], real_time_data: RTData):
        """
        Processa un messaggio che ha già matchato (fast path).
        """
        try:
            self._route_to_store(sub, value, real_time_data)
            self._subscription_service.mark_subscription_received(sub.id, "kafka")
        except Exception as e:
            logger.error(f"Error processing message for {sub.id}: {e}")
            self._subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))

    def _handle_message(self, topic: str, value: Dict[str, Any]):
        """
        Processa un messaggio Kafka (slow path per subscription senza ISIN filter).

        Args:
            topic: Topic Kafka di origine
            value: Messaggio deserializzato (dict)
        """
        subs = self._subscriptions_by_topic.get(topic, [])
        if not subs:
            return

        for sub in subs:
            # Skip se già processato nel fast path
            if sub.symbol_filter:
                continue

            try:
                if not sub.matches(value):
                    continue

                self._route_to_store(sub, value, self.real_time_data)
                self._subscription_service.mark_subscription_received(sub.id, "kafka")

            except Exception as e:
                logger.error(f"Error processing message for {sub.id}: {e}")
                self._subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))

    def _route_to_store(self, sub: KafkaSubscription, value: Dict[str, Any], real_time_data: RTData):
        """
        Routa i dati allo store appropriato.

        Args:
            sub: Subscription che ha matchato
            value: Messaggio deserializzato
            real_time_data: RTData instance (passato per evitare self lookup)
        """
        store = sub.store
        id_ = sub.id

        if store == "market":
            if sub.fields_mapping:
                data = sub.extract_fields(value)
            else:
                data = self._extract_default_market_fields(value)

            if data:
                real_time_data.update(id_, data, store="market")
                real_time_data.set_currency_for_id(id_, value.get("instrument", {}).get("currency"))

        elif store == "state":
            real_time_data.update(id_, value, store="state")

        elif store == "events":
            real_time_data._event_store.append(id_, value)

        elif store == "blob":
            real_time_data._blob_store.store(id_, value)

        elif store == "orders":
            from market_monitor.live_data_hub.order import Order
            source = sub.extract_fields(value) if sub.fields_mapping else value
            order = Order.from_dict(source)
            real_time_data.update_order(order)

    @staticmethod
    def _extract_default_market_fields(value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estrazione di default per formati comuni quando non c'è fields_mapping.

        Supporta:
        - BookBest (DUMA): bidBestLevel/askBestLevel
        - PublicDeal: price, quantity
        - Quote: bid, ask
        """
        data = {}

        # BookBest format (DUMA)
        if 'bidBestLevel' in value and value['bidBestLevel']:
            bid_level = value['bidBestLevel']
            if 'price' in bid_level:
                data['BID'] = float(bid_level['price'])
            if 'quantity' in bid_level:
                data['BID_SIZE'] = float(bid_level['quantity'])

        if 'askBestLevel' in value and value['askBestLevel']:
            ask_level = value['askBestLevel']
            if 'price' in ask_level:
                data['ASK'] = float(ask_level['price'])
            if 'quantity' in ask_level:
                data['ASK_SIZE'] = float(ask_level['quantity'])

        # PublicDeal format
        if 'price' in value:
            try:
                data['LAST'] = float(value['price'])
            except (ValueError, TypeError):
                pass
        if 'quantity' in value:
            try:
                data['SIZE'] = float(value['quantity'])
            except (ValueError, TypeError):
                pass

        # Generic quote format
        if 'bid' in value:
            try:
                data['BID'] = float(value['bid'])
            except (ValueError, TypeError):
                pass
        if 'ask' in value:
            try:
                data['ASK'] = float(value['ask'])
            except (ValueError, TypeError):
                pass

        return data


class KafkaOrderStreamingThread(_KafkaBaseThread):
    """
    Dedicated thread for streaming order data from Kafka.

    Consumes messages from order topics and routes each one to
    RTData.update_order() as an Order dataclass.  Only ACTIVE orders
    are kept in memory; EXPIRED / CANCELLED ones are automatically
    removed by the OrderStore.

    Subscriptions are registered via
        subscription_service.subscribe_orders_kafka(...)
    and are identified by ``sub.store == "orders"``.

    Args:
        real_time_data: RTData instance where orders will be stored
        subscription_service: Shared SubscriptionService (injected)
        bootstrap_servers: Kafka bootstrap servers string
        schema_registry_url: Avro Schema Registry URL
        start_mode: "latest" or "earliest"
        consumer_group: Kafka consumer group ID (default: random UUID)
    """

    def __init__(self,
                 real_time_data: RTData,
                 subscription_service: SubscriptionService,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 **kwargs):
        super().__init__(subscription_service, bootstrap_servers, schema_registry_url,
                         start_mode, consumer_group, **kwargs)
        self.name = "kafka_order"
        self.real_time_data = real_time_data

    @staticmethod
    def _is_order_subscription(sub: KafkaSubscription) -> bool:
        return sub.store == "orders"

    def _should_include_subscription(self, sub: KafkaSubscription) -> bool:
        return self._is_order_subscription(sub)

    def _process_message(self, topic: str, value: Dict[str, Any]):
        # FAST PATH: O(1) lookup by symbol_filter (typically ISIN)
        instrument = value.get("instrument")
        if instrument:
            topic_subs = self._isin_to_sub.get(topic, {})
            for field_value in instrument.values():
                sub = topic_subs.get(field_value)
                if sub:
                    self._process_order(sub, value)
                    break
            else:
                # Slow path: no symbol_filter or no match — try all topic subs
                for sub in self._subscriptions_by_topic.get(topic, []):
                    if not sub.symbol_filter and sub.matches(value):
                        self._process_order(sub, value)

    def _process_order(self, sub: KafkaSubscription, value: Dict[str, Any]):
        """Parse the raw Kafka message into an Order and route to RTData."""
        from market_monitor.live_data_hub.order import Order
        try:
            source = sub.extract_fields(value) if sub.fields_mapping else value
            order = Order.from_dict(source)
            self.real_time_data.update_order(order)
            self._subscription_service.mark_subscription_received(sub.id, "kafka")
        except Exception as e:
            logger.error(f"Error processing order for {sub.id}: {e}")
            self._subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))


class KafkaTradeStreamingThread(_KafkaBaseThread):
    """
    Thread standalone per streaming di dati di trade da Kafka (PublicDeal, Trade topics).

    Responsabilità (e solo queste):
    1. Riceve un SubscriptionService iniettato dall'esterno (DI).
    2. Legge le subscription Kafka dal SubscriptionService (pending + active).
    3. Scarta i messaggi non sottoscritti (fast-path O(1) per ISIN).
    4. Matcha i PublicDeal con i potenziali own-trade (deduplication).
    5. Mette le tuple (TradeType, pd.DataFrame) nella Queue condivisa.

    NON conosce RTData né route_to_store.

    Logica di deduplication own-trade vs market-trade:
    - Own trades emessi subito come (TradeType.OWN, df) e cancellano il primo
      public deal bufferizzato con la stessa chiave.
    - Public deals bufferizzati per `buffer_sec` secondi. Se arriva un own trade
      con la stessa chiave entro la finestra, il public deal viene scartato.
      Altrimenti viene emesso come (TradeType.MARKET, df) allo scadere del timer.

    TradeType è importato da trade.py (OWN=1, MARKET=2).

    Args:
        queue: Queue condivisa dove vengono messe le tuple (TradeType, pd.DataFrame)
        subscription_service: SubscriptionService iniettato, già popolato con le
            subscription Kafka di tipo trade (es. PublicDeal, Trade)
        bootstrap_servers: Lista server Kafka
        schema_registry_url: URL Schema Registry per deserializzazione Avro
        start_mode: "latest" (nuovi messaggi) o "earliest" (dall'inizio)
        consumer_group: Group ID per il consumer (default: UUID random)
        buffer_sec: Secondi di attesa prima di emettere un public deal come MARKET
        **kwargs: Parametri extra ignorati (per compatibilità con **kafka_params)
    """

    _TRADE_TOPIC_SUFFIXES = (".PublicDeal", ".Trade")

    def __init__(self,
                 queue: Queue,
                 subscription_service: SubscriptionService,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 buffer_sec: float = 1.0,
                 **kwargs):
        super().__init__(subscription_service, bootstrap_servers, schema_registry_url,
                         start_mode, consumer_group, **kwargs)
        self.name = "kafka_trade"
        self.queue_trade: Queue = queue

        # Deduplication state
        self._buffer_sec: float = buffer_sec
        self._pending_publicdeals: Dict[TradeKey, Deque[Any]] = defaultdict(deque)
        self._seen_own_trades: Set[TradeKey] = set()
        self._pending_trade_timestamp_received: Dict[TradeKey, float] = {}

    @classmethod
    def _is_trade_topic(cls, topic: str) -> bool:
        return any(topic.endswith(s) for s in cls._TRADE_TOPIC_SUFFIXES)

    def _should_include_subscription(self, sub: KafkaSubscription) -> bool:
        return self._is_trade_topic(sub.topic)

    def _on_poll_cycle(self):
        self._flush_public_deal_expired()

    def _process_message(self, topic: str, value: Dict[str, Any]):
        isin_to_sub = self._isin_to_sub.get(topic)
        # FAST PATH: O(1) lookup by ISIN
        instrument = value.get('instrument')
        if instrument:
            isin = instrument.get('isin')
            if isin:
                sub = isin_to_sub.get(isin) if isin_to_sub else None
                if sub:
                    try:
                        self._dispatch(sub, value)
                        self._subscription_service.mark_subscription_received(sub.id, "kafka")
                    except Exception as e:
                        logger.error(f"Error dispatching trade for {sub.id}: {e}")
                        self._subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))
                else:
                    topic_subs = self._subscriptions_by_topic.get(topic, [])
                    for sub in topic_subs:
                        if sub.matches(value):
                            self._dispatch(sub, value)
                            self._subscription_service.mark_subscription_received(sub.id, "kafka")

    # ------------------------------------------------------------------
    # Dispatch: estrazione campi + deduplication (no RTData.update)
    # ------------------------------------------------------------------

    def _dispatch(self, sub: KafkaSubscription, value: Dict[str, Any]):
        """Estrae i campi di trade e applica la deduplication, poi mette in queue."""
        data = self._extract_default_trade_fields(value)
        if sub.fields_mapping:
            data.update(sub.extract_fields(value))
        if data:
            self._handle_public_vs_own_deal(data)

    @staticmethod
    def _extract_default_trade_fields(value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estrae il set completo di campi da un messaggio PublicDeal o Trade.

        Campi richiesti da _handle_public_vs_own_deal():
        Isin, Market, Last Update, Price, Quantity, Own Trade, Ticker,
        Exchange, Currency, Description.
        """
        instrument = value.get('instrument') or {}
        side = value.get('side')
        if side == 'ASK':
            own_trade = -1
        elif side == 'BID':
            own_trade = 1
        else:
            own_trade = 0

        return {
            'ticker': instrument.get('symbol'),
            'isin': instrument.get('isin'),
            'currency': instrument.get('currency'),
            'market': instrument.get('market'),
            'exchange': instrument.get('market'),
            'description': None,
            'price': float(value['price']),
            'quantity': float(value['quantity']),
            'last_update': pd.to_datetime(value['eventTimestampUTC'], utc=True).tz_convert('Europe/Rome').tz_localize(
                None),
            'own_trade': own_trade,
        }

    # ------------------------------------------------------------------
    # Trade deduplication
    # ------------------------------------------------------------------

    def _flush_public_deal_expired(self):
        """
        Rilascia i public deal bufferizzati il cui tempo di attesa ha superato _buffer_sec.

        Chiamato ad ogni ciclo di poll (~100ms). I public deal non abbinati a un
        own trade entro la finestra vengono emessi come (TradeType.MARKET, df).
        """
        now = time.time()
        expired_keys = [
            key for key, ts in self._pending_trade_timestamp_received.items()
            if now - ts > self._buffer_sec
        ]
        for key in expired_keys:
            pending_queue = self._pending_publicdeals.pop(key, None)
            self._pending_trade_timestamp_received.pop(key, None)
            if not pending_queue:
                continue
            for df in pending_queue:
                self.queue_trade.put((TradeType.MARKET, df.set_index("ticker")))

    def _handle_public_vs_own_deal(self, data: Dict[str, Any]) -> None:
        """
        Classifica un messaggio di trade e applica la deduplication.

        La chiave di deduplication è (Isin, Market, Last Update, Price, Quantity).

        Own trades (Own Trade != 0):
            - Emessi subito come (TradeType.OWN, df)
            - Cancellano il primo public deal bufferizzato con la stessa chiave

        Public deals (Own Trade == 0):
            - Se la chiave è già in _seen_own_trades: scartati
            - Altrimenti: bufferizzati con timer di scadenza
        """
        key: TradeKey = (
            str(data['isin']),
            str(data['market']),
            data['last_update'].value,
            float(data['price']),
            float(data['quantity']),
        )
        is_own = data.get('own_trade', 0) != 0

        df = pd.DataFrame([data])

        if is_own:
            self.queue_trade.put((TradeType.OWN, df.set_index("ticker")))
            self.queue_trade.put((TradeType.MARKET, df.set_index("ticker")))
            self._seen_own_trades.add(key)
            # Cancella il primo public deal bufferizzato corrispondente
            if key in self._pending_publicdeals and self._pending_publicdeals[key]:
                self._pending_publicdeals[key].popleft()
                if not self._pending_publicdeals[key]:
                    self._pending_publicdeals.pop(key, None)
                    self._pending_trade_timestamp_received.pop(key, None)
        else:
            if key in self._seen_own_trades:
                # Già processato come own trade; scarta il public deal duplicato
                self._seen_own_trades.discard(key)
                return
            self._pending_publicdeals[key].append(df)
            if key not in self._pending_trade_timestamp_received:
                self._pending_trade_timestamp_received[key] = time.time()


# ============================================================================
# Main (esempio di utilizzo)
# ============================================================================

if __name__ == "__main__":
    import time
    from threading import Lock

    logging.basicConfig(level=logging.INFO)

    isin_to_subscribe = [
        "FEHY 26"
    ]

    # ------------------------------------------------------------------
    # Book thread: KafkaStreamingThread subscribes to BookBest topics
    # ------------------------------------------------------------------
    book_rtdata = RTData(locker=Lock(), fields=["BID", "ASK", "BID_SIZE", "ASK_SIZE"])
    book_svc = SubscriptionService()

    for isin in isin_to_subscribe:
        book_svc.subscribe_kafka(
            id=isin,
            topic="COALESCENT_DUMA.XEUR.Order",
            symbol_filter=isin,
            symbol_field="instrument.symbol",
            store="market"
        )

    book_thread = KafkaStreamingThread(book_rtdata, subscription_service=book_svc)
    book_thread.start()

    # ------------------------------------------------------------------
    # Trade thread: KafkaTradeStreamingThread — standalone, no RTData
    # Usa lo stesso SubscriptionService di RTData: filtra *.PublicDeal / *.Trade
    # ------------------------------------------------------------------
    # trade_sub_service = book_rtdata.get_subscription_manager()
    #
    # for isin in isin_to_subscribe:
    #     for topic in ["COALESCENT_DUMA.ETFP.PublicDeal", "COALESCENT_DUMA.ETFP.Trade"]:
    #         trade_sub_service.subscribe_kafka(
    #             id=f"{isin}_{topic.split('.')[-1]}",
    #             topic=topic,
    #             symbol_filter=isin,
    #         )
    #
    # trade_queue: Queue = Queue()
    # trade_thread = KafkaTradeStreamingThread(
    #     queue=trade_queue,
    #     subscription_service=trade_sub_service,
    #     buffer_sec=10.0,
    # )
    # trade_thread.start()

    # ------------------------------------------------------------------
    # Monitor both streams
    # ------------------------------------------------------------------
    print("Monitoring ETF book prices and trade queue (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(10)

            print("\n=== Book Market Data ===")
            book_data = book_rtdata.get_data_field()
            if not book_data.empty:
                print(book_data)


    except KeyboardInterrupt:
        book_thread.stop()

        book_thread.join(timeout=5)

        print("Stopped")
