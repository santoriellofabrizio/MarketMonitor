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
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Tuple, Deque
from collections import defaultdict, deque
from queue import Queue
import pandas as pd
import time

from market_monitor.input_threads.trade import TradeType
from market_monitor.live_data_hub.real_time_data_hub import RTData
from market_monitor.live_data_hub.live_subscription import KafkaSubscription

logger = logging.getLogger(__name__)
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

    DEFAULT_BOOTSTRAP_SERVERS = "aftstserver51.af.tst:9092,aftstserver52.af.tst:9092,aftstserver53.af.tst:9092"
    DEFAULT_SCHEMA_REGISTRY = "http://aftstserver51.af.tst:8081,http://aftstserver52.af.tst:8081,http://aftstserver53.af.tst:8081"

    def __init__(self,
                 real_time_data: RTData,
                 bootstrap_servers: Optional[str] = None,
                 schema_registry_url: Optional[str] = None,
                 start_mode: str = "latest",
                 consumer_group: Optional[str] = None,
                 q_trade: Queue = Queue(),
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

        # queue
        self.queue_trade: Queue = q_trade
        self._buffer_sec: float = 1  # Wait this seconds before insert a market trade
        self._pending_publicdeals: Dict[TradeKey, Deque[pd.DataFrame]] = defaultdict(deque)
        self._seen_own_trades: Set[TradeKey] = set()
        self._pending_trade_timestamp_received: Dict[TradeKey, float] = {}

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
                self._flush_public_deal_expired()

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

    def _flush_public_deal_expired(self):
        """
        Check for every public deal already received:
        1. First check all expired trade key by iterating self._pending_trade_timestamp_received, which is a dictionary
           with TradeKey as key and timestamp as value (note that there could be more than one trade with the same
           TradeKey).
        2. Check all the expired keys, pop the TradeKey from self._pending_publicdeals, which is a dictionary with
           TradeKey as key and a deque of dictionary (it could be more than one trade for TradeKey).
        3. Pop all the expired keys in the two dictionaries.
        4. Push all the trade in the popped deque as MarketTrade.
        """
        now = time.time()
        expired_keys = [
            key for key, ts in self._pending_trade_timestamp_received.items()
            if now - ts > self._buffer_sec
        ]
        for key in expired_keys:
            queue = self._pending_publicdeals.pop(key, None)
            self._pending_trade_timestamp_received.pop(key, None)
            if not queue:
                continue
            for data in queue:
                self.queue_trade.put((TradeType.MARKET, data))

    def _handle_public_vs_own_deal(self, data: Dict[str, Any]) -> bool:
        """
        Method to handle both market and own_deal.
        Whenever we received a trade data:
        1. Create a TradeKey (it is not unique, because we can have multiple trades with same timestamp, price, quantity).
        2. If the trade is a bsh trade:
           2.1. push in the queue as own trade and add the trade key in self._seen_own_trades;
           2.2. check if a trade with the same key is present in self._pending_publicdeals;
           2.3. if is present, pop only one trade of the queue in self._pending_publicdeals[TradeKey];
        3. If the trade is a public deal:
           3.1. if there is an identical trade in self._seen_own_trades, discard the public deal and also discard
                the own trade saved in self._seen_own_trades (it's been already pushed so no worries);
           3.2. if there isn't an identical trade in self._seen_own_trades, put the public deal in the buffer
                self._pending_publicdeals[TradeKey];
           3.3. if it's the first trade with this TradeKey, save the timestamp received in self._pending_trade_timestamp_received[tradeKey].
        """
        key: TradeKey = (
            str(data["isin"]),
            str(data["market"]),
            int(data["last_update_int"]),
            data["price"],
            data["quantity"]
        )
        is_own = data.get("own_trade", 0) != 0
        data = pd.DataFrame([data])[['ticker', 'isin', 'currency', 'quantity', 'price', 'last_update', 'exchange', 'market', 'description',"own_trade"]]
        if is_own:
            self.queue_trade.put((TradeType.OWN, data))
            self._seen_own_trades.add(key)
            if key in self._pending_publicdeals and self._pending_publicdeals[key]:
                self._pending_publicdeals[key].popleft()
                if not self._pending_publicdeals[key]:
                    self._pending_publicdeals.pop(key, None)
                    self._pending_trade_timestamp_received.pop(key, None)
        else:
            if key in self._seen_own_trades:
                self._seen_own_trades.discard(key)
                return False

            self._pending_publicdeals[key].append(data)
            if key not in self._pending_trade_timestamp_received:
                self._pending_trade_timestamp_received[key] = time.time()
        return True

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
                continue_iteration = self._handle_public_vs_own_deal(data)
                if not continue_iteration:
                    return
                real_time_data.update(ticker, data, store="market")

        elif store == "state":
            real_time_data.update(ticker, value, store="state")

        elif store == "events":
            real_time_data._event_store.append(ticker, value)

        elif store == "blob":
            real_time_data._blob_store.store(ticker, value)

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

        if 'instrument' in value:
            if 'symbol' in value['instrument']:
                try:
                    data['ticker'] = value['instrument']['symbol']
                except (ValueError, TypeError):
                    pass
            if 'isin' in value['instrument']:
                try:
                    data['isin'] = value['instrument']['isin']
                except (ValueError, TypeError):
                    pass
            if 'market' in value['instrument']:
                try:
                    data['market'] = value['instrument']['market']
                    data['exchange'] = value['instrument']['market']
                except (ValueError, TypeError):
                    pass
            if 'currency' in value['instrument']:
                try:
                    data['currency'] = value['instrument']['currency']
                except (ValueError, TypeError):
                    pass
            data['description'] = None

        if 'price' in value:
            try:
                data['price'] = float(value['price'])
            except (ValueError, TypeError):
                pass
        if 'quantity' in value:
            try:
                data['quantity'] = float(value['quantity'])
            except (ValueError, TypeError):
                pass

        if 'eventTimestampUTC' in value:
            try:
                data['last_update_int'] = (ts := value['eventTimestampUTC'])
                data['last_update'] = datetime.fromtimestamp(ts / 1_000_000_000)
            except (ValueError, TypeError):
                pass

        if 'side' in value:
            try:
                data['own_trade'] = -1 if value["side"] == "ASK" else +1
            except (ValueError, TypeError):
                pass
        else:
            data['own_trade'] = 0

        return data

    def stop(self):
        """
        Stop the thread.
        """
        self.stop_event.set()
        logger.info("KafkaStreamingThread stop requested")


# ============================================================================
# Main (esempio di utilizzo)
# ============================================================================

if __name__ == "__main__":
    import time
    from threading import Lock

    logging.basicConfig(level=logging.INFO)

    lock = Lock()
    rtdata = RTData(locker=lock, fields=["LAST", "SIZE", "TIMESTAMP"])

    isin_to_subscribe = [ "IE00BKM4GZ66", "IE00B4L5YC18", "IE00BP3QZ601", "LU0950668870", "IE00B0M63516",
                          "LU1900068914", "LU0659579733", "LU1781541252", "IE00B469F816", "LU0779800910",
                          "IE00BP3QZ825", "IE00B4L5YX21", "IE00B5L8K969", "IE00B02KXH56", "IE00BZCQB185",
                          "FR0010429068", "LU0514695690", "IE00B4K48X80", "LU0950674175", "LU0480132876",
                          "IE00099GAJC6", "LU0846194776", "IE00B6R52259", "LU0147308422", "LU1900066207",
                          "LU1900067940", "DE000A0Q4R85", "IE000Y77LGG9", "FR0014003IY1", "IE00BMY76136",
                          "LU0274209740", "FR0010245514", "LU1681043599", "IE00BHZPJ783", "LU2573967036",
                          "IE00BCHWNQ94", "IE00B0M63177", "FR0010361683", "IE00B44Z5B48", "IE00BHZRR147",
                          "LU2376679564", "IE00BP3QZB59", "FR0010315770", "IE00BFNM3J75", "LU1681044480",
                          "IE00B60SX394", "IE00BKX55T58", "IE00BTJRMP35", "LU0274209237", "IE000UQND7H4",
                          "IE00B945VV12", "LU2573966905", "IE00BZ02LR44"]
    thread_to_subscribe = ["COALESCENT_DUMA.ETFP.PublicDeal", "COALESCENT_DUMA.ETFP.Trade"]
    svc = rtdata.get_subscription_manager()
    for isin in isin_to_subscribe:
        for topic in thread_to_subscribe:
            svc.subscribe_kafka(
                id=f"{isin}_{topic.split('.')[-1]}",
                topic=topic,
                symbol_filter=isin,
                symbol_field="instrument.isin",
            )

    # Start thread
    kafka_thread = KafkaStreamingThread(rtdata)
    kafka_thread.start()

    # Monitor
    print("Monitoring ETF prices (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(10)
            data_retrieved = rtdata.get_data_field()
            if not data_retrieved.empty:
                print(f"\n=== Market Data ===")
                print(data_retrieved)
    except KeyboardInterrupt:
        kafka_thread.stop()
        kafka_thread.join(timeout=5)
        print("Stopped")
