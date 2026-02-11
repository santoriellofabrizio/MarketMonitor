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
import uuid
from typing import Optional, Dict, Any, List, Set

from market_monitor.live_data_hub.real_time_data_hub import RTData
from market_monitor.live_data_hub.live_subscription import KafkaSubscription

logger = logging.getLogger(__name__)


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
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 **kwargs):

        super().__init__(daemon=True)
        self.name = "kafka"

        # RTData reference
        self.real_time_data = real_time_data
        self.subscription_service = real_time_data.get_subscription_manager()

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
        self._subscriptions_by_topic: Dict[str, List[KafkaSubscription]] = {}
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
                    isin = instrument.get('isin')
                    if isin:
                        sub = isin_to_sub.get(isin)
                        if sub:
                            self._process_matched_message(sub, value, real_time_data)
                            continue
                
                # SLOW PATH: fallback per subscription senza symbol_filter
                topic = msg.topic()
                self._handle_message(topic, value)

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
        ticker = sub.id

        if store == "market":
            # Extract fields using subscription's mapping
            if sub.fields_mapping:
                data = sub.extract_fields(value)
            else:
                # Default extraction for common formats
                data = self._extract_default_market_fields(value)

            if data:
                real_time_data.update(ticker, data, store="market")

        elif store == "state":
            real_time_data.update(ticker, value, store="state")

        elif store == "events":
            real_time_data._event_store.append(ticker, value)

        elif store == "blob":
            real_time_data._blob_store.store(ticker, value)

    def _extract_default_market_fields(self, value: Dict[str, Any]) -> Dict[str, float]:
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


# ============================================================================
# Main (esempio di utilizzo)
# ============================================================================

if __name__ == "__main__":
    import time
    from threading import Lock

    logging.basicConfig(level=logging.INFO)

    # Setup
    lock = Lock()
    rtdata = RTData(locker=lock, fields=["BID", "ASK", "BID_SIZE", "ASK_SIZE"])

    # Subscribe a ETF via subscription service
    svc = rtdata.get_subscription_manager()

    # IWDA - iShares Core MSCI World
    svc.subscribe_kafka(
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

    # CSPX - iShares Core S&P 500
    svc.subscribe_kafka(
        id="CSPX",
        topic="COALESCENT_DUMA.ETFP.BookBest",
        symbol_filter="IE00B5BMR087",
        symbol_field="instrument.isin",
        store="market"
        # No fields_mapping -> uses default extraction
    )

    # Start thread
    kafka_thread = KafkaStreamingThread(rtdata)
    kafka_thread.start()

    # Monitor
    print("Monitoring ETF prices (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(2)
            data = rtdata.get_data_field()
            print(f"\n=== Market Data ===")
            print(data)
    except KeyboardInterrupt:
        kafka_thread.stop()
        kafka_thread.join(timeout=5)
        print("Stopped")
