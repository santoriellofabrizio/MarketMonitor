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


class KafkaStreamingThread(threading.Thread):
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

    # Default configuration (cluster test Sella)
    DEFAULT_BOOTSTRAP_SERVERS = "aftstserver51.af.tst:9092,aftstserver52.af.tst:9092,aftstserver53.af.tst:9092"
    DEFAULT_SCHEMA_REGISTRY = "http://aftstserver51.af.tst:8081,http://aftstserver52.af.tst:8081,http://aftstserver53.af.tst:8081"

    def __init__(self,
                 real_time_data: RTData,
                 subscription_service: SubscriptionService,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 **kwargs):

        super().__init__(daemon=True)
        self.symbol_filter_by_topic: dict = defaultdict(dict)
        self.name = "kafka"

        # RTData reference
        self.real_time_data = real_time_data
        self.subscription_service = subscription_service

        # Kafka configuration
        self.bootstrap_servers = bootstrap_servers or self.DEFAULT_BOOTSTRAP_SERVERS
        self.schema_registry_url = schema_registry_url or self.DEFAULT_SCHEMA_REGISTRY
        self.start_mode = start_mode
        self.consumer_group = consumer_group or str(uuid.uuid4())

        # State
        self.stop_event = threading.Event()
        self.running = False
        self.consumer = None
        self.avro_deserializer = None

        # Subscriptions indexed by topic for fast lookup
        self._subscriptions_by_topic = {}
        self._topics: Set[str] = set()

        # FAST LOOKUP: isin -> (subscription_id, KafkaSubscription) per O(1) match
        self._isin_to_sub: Dict[str, KafkaSubscription] = {}

    def _load_subscriptions(self):
        """Carica le sottoscrizioni Kafka dal SubscriptionService."""
        # Get pending + active subscriptions
        pending = self.subscription_service.get_pending_subscriptions("kafka") or {}
        active = self.subscription_service.get_kafka_subscription() or {}
        all_subs = {**pending, **active}

        # Reset
        self._subscriptions_by_topic.clear()
        self._topics.clear()

        for sub_id, sub in all_subs.items():
            if isinstance(sub, KafkaSubscription):
                topic = sub.topic
                if topic not in self._subscriptions_by_topic:
                    self._subscriptions_by_topic[topic] = []
                self._subscriptions_by_topic[topic].append(sub)
                self._topics.add(topic)

                # Build fast ISIN lookup
                if sub.symbol_filter:
                    self._isin_to_sub[sub.symbol_filter] = sub

        logger.info(
            f"KafkaStreamingThread: loaded {len(all_subs)} subscriptions "
            f"for {len(self._topics)} topics: {self._topics}"
        )

    def run(self):
        """Loop principale: connetti a Kafka e processa messaggi."""
        try:
            from confluent_kafka import Consumer
            from confluent_kafka.schema_registry import SchemaRegistryClient
            from confluent_kafka.schema_registry.avro import AvroDeserializer
        except ImportError as e:
            logger.error(f"Kafka libraries not installed: {e}")
            logger.error("Install with: pip install confluent_kafka fastavro")
            return

        # Load subscriptions
        self._load_subscriptions()

        if not self._topics:
            logger.warning("No Kafka topics to subscribe to, thread will idle")
            # Stay alive but idle, in case subscriptions are added later
            while not self.stop_event.wait(timeout=5):
                self._load_subscriptions()
                if self._topics:
                    break
            if not self._topics:
                logger.info("KafkaStreamingThread stopping (no subscriptions)")
                return

        # Setup Schema Registry
        schema_registry_conf = {"url": self.schema_registry_url}
        schema_registry_client = SchemaRegistryClient(schema_registry_conf)
        self.avro_deserializer = AvroDeserializer(schema_registry_client=schema_registry_client)

        # Setup Consumer
        consumer_conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.consumer_group,
            "auto.offset.reset": self.start_mode,
        }
        self.consumer = Consumer(consumer_conf)
        self.consumer.subscribe(list(self._topics))

        logger.info(f"KafkaStreamingThread started - topics: {self._topics}")
        self.running = True

        # Cache locals for speed in hot loop
        isin_to_sub = self._isin_to_sub
        real_time_data = self.real_time_data
        deserializer = self.avro_deserializer
        stop_event = self.stop_event
        consumer = self.consumer

        for tpc in self._topics:
            self.symbol_filter_by_topic[tpc] = {sub.symbol_filter: sub for sub in self._subscriptions_by_topic[tpc]}

        try:
            while not stop_event.is_set():
                # Poll for messages (0.1s = più reattivo)
                msg = consumer.poll(timeout=0.1)
                if msg is None:
                    # Check for new subscriptions periodically
                    self._check_new_subscriptions()
                    continue

                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                # Deserialize and process
                try:
                    value = deserializer(msg.value())
                except Exception as e:
                    logger.debug(f"Failed to deserialize message: {e}")
                    continue

                # FAST PATH: O(1) lookup by ISIN
                instrument = value.get('instrument')
                if instrument:
                    subs = self.symbol_filter_by_topic.get(msg.topic())
                    for k, v in instrument.items():
                        if sub := subs.get(v):
                            self._process_matched_message(sub, value, real_time_data)


        except Exception as e:
            logger.error(f"Error in KafkaStreamingThread: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
            self.running = False
            logger.info("KafkaStreamingThread stopped")

    def _check_new_subscriptions(self):
        """Check for new subscriptions and update consumer if needed."""
        pending = self.subscription_service.get_pending_subscriptions("kafka") or {}

        new_topics = set()
        for sub_id, sub in pending.items():
            if isinstance(sub, KafkaSubscription) and sub.topic not in self._topics:
                new_topics.add(sub.topic)

        if new_topics:
            logger.info(f"New Kafka topics detected: {new_topics}")
            self._load_subscriptions()
            if self.consumer:
                self.consumer.subscribe(list(self._topics))

    def _process_matched_message(self, sub: KafkaSubscription, value: Dict[str, Any], real_time_data: RTData):
        """
        Processa un messaggio che ha già matchato (fast path).
        """
        try:
            # Extract and route to store
            self._route_to_store(sub, value, real_time_data)

            # Mark subscription as received
            self.subscription_service.mark_subscription_received(sub.id, "kafka")
        except Exception as e:
            logger.error(f"Error processing message for {sub.id}: {e}")
            self.subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))

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
                self.subscription_service.mark_subscription_received(sub.id, "kafka")

            except Exception as e:
                logger.error(f"Error processing message for {sub.id}: {e}")
                self.subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))

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
            # Extract fields using subscription's mapping
            if sub.fields_mapping:
                data = sub.extract_fields(value)
            else:
                # Default extraction for common formats
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

    def stop(self):
        """Ferma il thread in modo pulito."""
        self.stop_event.set()
        logger.info("KafkaStreamingThread stop requested")


class KafkaTradeStreamingThread(threading.Thread):
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

    DEFAULT_BOOTSTRAP_SERVERS = KafkaStreamingThread.DEFAULT_BOOTSTRAP_SERVERS
    DEFAULT_SCHEMA_REGISTRY = KafkaStreamingThread.DEFAULT_SCHEMA_REGISTRY

    def __init__(self,
                 queue: Queue,
                 subscription_service: SubscriptionService,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 buffer_sec: float = 1.0,
                 **kwargs):
        super().__init__(daemon=True)
        self.name = "kafka_trade"

        # Output
        self.queue_trade: Queue = queue

        # SubscriptionService iniettato — unica fonte di verità per le subscription
        self._subscription_service: SubscriptionService = subscription_service

        # Kafka config
        self.bootstrap_servers = bootstrap_servers or self.DEFAULT_BOOTSTRAP_SERVERS
        self.schema_registry_url = schema_registry_url or self.DEFAULT_SCHEMA_REGISTRY
        self.start_mode = start_mode
        self.consumer_group = consumer_group or str(uuid.uuid4())

        # State
        self.stop_event = threading.Event()
        self.running = False
        self.consumer = None
        self.avro_deserializer = None

        # Subscription indexes (fast O(1) ISIN lookup)
        self._subscriptions_by_topic: Dict[str, List[KafkaSubscription]] = {}
        self._topics: Set[str] = set()
        self._isin_to_sub: defaultdict[str, Dict[str, KafkaSubscription]] = defaultdict(dict)

        # Deduplication state
        self._buffer_sec: float = buffer_sec
        self._pending_publicdeals: Dict[TradeKey, Deque[Any]] = defaultdict(deque)
        self._seen_own_trades: Set[TradeKey] = set()
        self._pending_trade_timestamp_received: Dict[TradeKey, float] = {}

    _TRADE_TOPIC_SUFFIXES = (".PublicDeal", ".Trade")

    @classmethod
    def _is_trade_topic(cls, topic: str) -> bool:
        return any(topic.endswith(s) for s in cls._TRADE_TOPIC_SUFFIXES)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def _load_subscriptions(self):
        """
        Legge le subscription Kafka dal SubscriptionService condiviso e indicizza
        solo i topic di trade (*.PublicDeal, *.Trade).
        """
        pending = self._subscription_service.get_pending_subscriptions("kafka") or {}
        active = self._subscription_service.get_kafka_subscription() or {}
        all_subs = {**pending, **active}

        self._subscriptions_by_topic.clear()
        self._topics.clear()
        self._isin_to_sub.clear()

        count = 0
        for sub in all_subs.values():
            if isinstance(sub, KafkaSubscription) and self._is_trade_topic(sub.topic):
                topic = sub.topic
                if topic not in self._subscriptions_by_topic:
                    self._subscriptions_by_topic[topic] = []
                self._subscriptions_by_topic[topic].append(sub)
                self._topics.add(topic)
                if sub.symbol_filter:
                    self._isin_to_sub[topic][sub.symbol_filter] = sub
                count += 1

        logger.info(
            f"KafkaTradeStreamingThread: loaded {count} trade subscriptions "
            f"for {len(self._topics)} topics: {self._topics}"
        )

    def _check_new_subscriptions(self):
        """Controlla se ci sono nuove subscription pending di tipo trade e aggiorna il consumer."""
        pending = self._subscription_service.get_pending_subscriptions("kafka") or {}
        new_topics = {
            sub.topic
            for sub in pending.values()
            if isinstance(sub, KafkaSubscription)
               and self._is_trade_topic(sub.topic)
               and sub.topic not in self._topics
        }
        if new_topics:
            logger.info(f"KafkaTradeStreamingThread: new trade topics detected: {new_topics}")
            self._load_subscriptions()
            if self.consumer:
                self.consumer.subscribe(list(self._topics))

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self):
        """Loop di polling: connette a Kafka e invia i trade in queue."""
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
            logger.warning("KafkaTradeStreamingThread: no topics, thread idle")
            while not self.stop_event.wait(timeout=5):
                self._load_subscriptions()
                if self._topics:
                    break
            if not self._topics:
                logger.info("KafkaTradeStreamingThread stopping (no subscriptions)")
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

        logger.info(f"KafkaTradeStreamingThread started - topics: {self._topics}")
        self.running = True

        # Cache locals for speed in hot loop
        deserializer = self.avro_deserializer
        stop_event = self.stop_event
        consumer = self.consumer

        try:
            while not stop_event.is_set():
                msg = consumer.poll(timeout=0.1)

                # Flush expired buffered public deals on every poll cycle
                self._flush_public_deal_expired()

                if msg is None:
                    self._check_new_subscriptions()
                    continue

                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                try:
                    value = deserializer(msg.value())
                except Exception as e:
                    logger.debug(f"Failed to deserialize trade message: {e}")
                    continue

                isin_to_sub = self._isin_to_sub.get(msg.topic())
                # FAST PATH: O(1) lookup by ISIN
                instrument = value.get('instrument')
                if instrument:
                    isin = instrument.get('isin')
                    if isin:
                        sub = isin_to_sub.get(isin)
                        if sub:
                            try:
                                self._dispatch(sub, value)
                                self._subscription_service.mark_subscription_received(sub.id, "kafka")
                            except Exception as e:
                                logger.error(f"Error dispatching trade for {sub.id}: {e}")
                                self._subscription_service.mark_subscription_failed(sub.id, "kafka", str(e))
                        else:
                            topic_subs = self._subscriptions_by_topic.get(msg.topic(), [])
                            for sub in topic_subs:
                                if sub.matches(value):
                                    self._dispatch(sub, value)
                                    self._subscription_service.mark_subscription_received(sub.id, "kafka")

        except Exception as e:
            logger.error(f"Error in KafkaTradeStreamingThread: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
            self.running = False
            logger.info("KafkaTradeStreamingThread stopped")

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

    def stop(self):
        """Ferma il thread in modo pulito."""
        self.stop_event.set()
        logger.info("KafkaTradeStreamingThread stop requested")


# ============================================================================
# Main (esempio di utilizzo)
# ============================================================================

if __name__ == "__main__":
    import time
    from threading import Lock

    logging.basicConfig(level=logging.INFO)

    isin_to_subscribe = [
        "IE00BKM4GZ66", "IE00B4L5YC18", "IE00BP3QZ601", "LU0950668870", "IE00B0M63516",
        "LU1900068914", "LU0659579733", "LU1781541252", "IE00B469F816", "LU0779800910",
        "IE00BP3QZ825", "IE00B4L5YX21", "IE00B5L8K969", "IE00B02KXH56", "IE00BZCQB185",
        "FR0010429068", "LU0514695690", "IE00B4K48X80", "LU0950674175", "LU0480132876",
        "IE00099GAJC6", "LU0846194776", "IE00B6R52259", "LU0147308422", "LU1900066207",
        "LU1900067940", "DE000A0Q4R85", "IE000Y77LGG9", "FR0014003IY1", "IE00BMY76136",
        "LU0274209740", "FR0010245514", "LU1681043599", "IE00BHZPJ783", "LU2573967036",
        "IE00BCHWNQ94", "IE00B0M63177", "FR0010361683", "IE00B44Z5B48", "IE00BHZRR147",
        "LU2376679564", "IE00BP3QZB59", "FR0010315770", "IE00BFNM3J75", "LU1681044480",
        "IE00B60SX394", "IE00BKX55T58", "IE00BTJRMP35", "LU0274209237", "IE000UQND7H4",
        "IE00B945VV12", "LU2573966905", "IE00BZ02LR44",
    ]

    # ------------------------------------------------------------------
    # Book thread: KafkaStreamingThread subscribes to BookBest topics
    # ------------------------------------------------------------------
    book_rtdata = RTData(locker=Lock(), fields=["BID", "ASK", "BID_SIZE", "ASK_SIZE"])
    book_svc = book_rtdata.get_subscription_manager()

    for isin in isin_to_subscribe:
        book_svc.subscribe_kafka(
            id=isin,
            topic="COALESCENT_DUMA.ETFP.BookBest",
            symbol_filter=isin,
            symbol_field="instrument.isin",
            store="market",
            fields_mapping={
                "BID": "bidBestLevel.price",
                "ASK": "askBestLevel.price",
                "BID_SIZE": "bidBestLevel.quantity",
                "ASK_SIZE": "askBestLevel.quantity",
            }
        )

    book_thread = KafkaStreamingThread(book_rtdata)
    book_thread.start()

    # ------------------------------------------------------------------
    # Trade thread: KafkaTradeStreamingThread — standalone, no RTData
    # Usa lo stesso SubscriptionService di RTData: filtra *.PublicDeal / *.Trade
    # ------------------------------------------------------------------
    trade_sub_service = book_rtdata.get_subscription_manager()

    for isin in isin_to_subscribe:
        for topic in ["COALESCENT_DUMA.ETFP.PublicDeal", "COALESCENT_DUMA.ETFP.Trade"]:
            trade_sub_service.subscribe_kafka(
                id=f"{isin}_{topic.split('.')[-1]}",
                topic=topic,
                symbol_filter=isin,
            )

    trade_queue: Queue = Queue()
    trade_thread = KafkaTradeStreamingThread(
        queue=trade_queue,
        subscription_service=trade_sub_service,
        buffer_sec=10.0,
    )
    trade_thread.start()

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

            print("\n=== Trades ===")
            while not trade_queue.empty():
                trade_type, df = trade_queue.get_nowait()
                print(f"[{trade_type.name}]")
                print(df.to_string(index=False))

    except KeyboardInterrupt:
        book_thread.stop()
        trade_thread.stop()
        book_thread.join(timeout=5)
        trade_thread.join(timeout=5)
        print("Stopped")
