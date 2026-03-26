"""
Worker threads per TradeDashboard - VERSIONE AGGIORNATA.
Gestiscono polling dati da Queue o RedisPublisher Pub/Sub.

AGGIORNAMENTO: Supporto completo per formato flat RTD-compatible.
"""
import json
import logging
import traceback
from collections import deque

from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
from typing import Dict, Any, Tuple
from queue import Empty
import time

from market_monitor.gui.implementations.PyQt5Dashboard.common import safe_concat

logger = logging.getLogger(__name__)


# ============================================================================
# Parser Universale (embedded)
# ============================================================================

def parse_redis_message(json_str: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Parse messaggio RedisPublisher in formato flat o wrapped.

    Returns:
        Tuple (data, metadata)
    """
    try:
        if isinstance(json_str, str):
            msg = json.loads(json_str)
        else:
            msg = json_str  # Già dict (da Queue)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e} — raw={repr(json_str)[:200]}")
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
        return msg, {}


# ============================================================================
# CLASSE 1: Base astratta
# ============================================================================
class BaseDashboardThread(QThread):
    """Base class per worker threads della dashboard"""

    # Signals comuni
    data_updated = pyqtSignal(pd.DataFrame)
    status_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False

    @staticmethod
    def _to_dataframe(data):
        """Convert data to DataFrame"""
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:
                logger.debug("_to_dataframe: received empty list, returning None")
                return None
            df = pd.DataFrame(data)
        elif isinstance(data, str):
            try:
                df = pd.DataFrame.from_records(json.loads(data))
            except Exception as e:
                logger.error(f"_to_dataframe: failed to parse string as JSON records: {e} — data={repr(data)[:200]}")
                return None
        else:
            logger.warning(f"_to_dataframe: unsupported data type {type(data).__name__}, returning None")
            return None

        if df.empty:
            logger.debug("_to_dataframe: resulting DataFrame is empty")
            return df

        # CONVERSIONE TIMESTAMP: Converti colonne timestamp in datetime pandas
        timestamp_cols = ['timestamp', 'time', 'datetime', 'last_update']
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    before_nulls = df[col].isna().sum()
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    after_nulls = df[col].isna().sum()
                    if after_nulls > before_nulls:
                        logger.warning(
                            f"_to_dataframe: {after_nulls - before_nulls} values in '{col}' "
                            f"could not be parsed as datetime"
                        )
                except Exception as e:
                    logger.error(f"_to_dataframe: unexpected error converting '{col}' to datetime: {e}")

        logger.debug(f"_to_dataframe: produced DataFrame shape={df.shape}, columns={list(df.columns)}")
        return df

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait(2000)


# ============================================================================
# CLASSE 2: Queue Polling Thread - AGGIORNATO
# ============================================================================
class QueuePollingThread(BaseDashboardThread):
    """
    Worker thread per Queue mode.
    Legge messaggi dict da Queue Python.

    AGGIORNAMENTO: Supporta formato flat e wrapped.

    Formato messaggi attesi:
        Wrapped: {'type': 'data', 'data': df}
        Flat: {"AAPL": 150, "__metadata__": {"type": "DATA"}}
    """

    def __init__(self, datasource):
        super().__init__()
        self.datasource = datasource
        # Accesso diretto alla queue interna
        self.queue = datasource._queue

    def run(self):
        """Main loop - polling dalla queue"""
        self.running = True
        logger.info("[QueuePolling] Thread started")

        while self.running:
            try:
                # Blocking get con timeout dalla queue
                message = self.queue.get(timeout=0.05)

                logger.debug(f"[QueuePolling] Received message type={type(message).__name__}")
                self._process_message_universal(message)

            except Empty:
                # Nessun messaggio, continua
                continue

            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"[QueuePolling] Unexpected error: {e}\n{error_detail}")
                self.error_occurred.emit(f"Queue polling error: {e}")
                time.sleep(0.1)

        logger.info("[QueuePolling] Thread stopped")

    def _process_message_universal(self, message):
        """
        Processa messaggio con parser universale.
        Supporta flat e wrapped format.

        NUOVO METODO che sostituisce _process_message.
        """
        if not isinstance(message, dict):
            self.error_occurred.emit(f"Invalid message format: {type(message)}")
            return

        try:
            # Parse universale
            data, metadata = parse_redis_message(message)
            msg_type = metadata.get('type', message.get('type', 'unknown'))

            # Processa basato su tipo
            if msg_type == 'data' or msg_type == 'DATA':
                self._handle_data(data, metadata)

            elif msg_type in ('status', 'flow_detected', 'command', 'config'):
                self._handle_status(msg_type, data, metadata)

            elif msg_type == 'error':
                error_msg = metadata.get('message', data if isinstance(data, str) else 'Unknown error')
                self.error_occurred.emit(str(error_msg))

            else:
                # Tipo sconosciuto: tratta come dati
                logger.warning(f"[QueuePolling] Unknown message type: '{msg_type}', treating as data")
                self._handle_data(data, metadata)

        except Exception as e:
            self.error_occurred.emit(f"Message processing error: {str(e)}")

    def _handle_data(self, data: Any, metadata: Dict):
        """Gestisce messaggi di tipo data"""
        try:
            df = self._to_dataframe(data)
            if df is None:
                logger.warning("[QueuePolling] _handle_data: _to_dataframe returned None, skipping emit")
                return
            logger.debug(f"[QueuePolling] Emitting data_updated: {len(df)} rows")
            self.data_updated.emit(df)
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"[QueuePolling] Data conversion error: {e}\n{error_detail}")
            self.error_occurred.emit(f"Data conversion error: {e}")

    def _handle_status(self, msg_type: str, data: Any, metadata: Dict):
        """Gestisce messaggi di status/command/flow"""
        status_data = {
            'type': msg_type,
            'data': data
        }
        # Merge metadata se presente
        if metadata:
            status_data.update(metadata)

        self.status_updated.emit(status_data)

    def _process_message(self, message: dict):
        """
        LEGACY: Processa messaggio wrapped.
        Mantenuto per backward compatibility.

        Ora usa internamente _process_message_universal.
        """
        self._process_message_universal(message)


# ============================================================================
# CLASSE 3: RedisPublisher Pub/Sub Thread - AGGIORNATO
# ============================================================================
class RedisPubSubThread(BaseDashboardThread):
    """
    Worker thread per RedisPublisher mode.
    Sottoscrive RedisPublisher Pub/Sub channel.

    AGGIORNAMENTO: Parser universale per flat/wrapped format.
    """

    def __init__(self, redis_config: Dict):
        super().__init__()
        self.batch_queue = deque()
        self.last_emit_time = 0
        self.batch_interval = redis_config.get('batch_interval', 0.5) # 500ms
        self.batch_max_size = None

        self.redis_config = redis_config
        self.redis_client = None
        self.pubsub = None

    def run(self):
        """Main loop — Redis PubSub listener with automatic reconnection."""
        try:
            import redis
            from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
        except ImportError:
            self.error_occurred.emit("Redis not installed. Run: pip install redis")
            return

        self.running = True
        host = self.redis_config.get('host', 'localhost')
        port = self.redis_config.get('port', 6379)
        db = self.redis_config.get('db', 0)
        channel = self.redis_config.get('channel', 'trades_df')

        max_retries = self.redis_config.get('max_retries', 10)  # 0 = unlimited
        retry_delay = self.redis_config.get('retry_delay', 2.0)  # initial sleep (s)
        max_delay = self.redis_config.get('max_retry_delay', 30.0)
        attempt = 0

        logger.info(f"[RedisPubSub] Thread started — {host}:{port} db={db} channel={channel}")

        while self.running:
            try:
                # ── Connect ──────────────────────────────────────────────
                self.redis_client = redis.Redis(
                    host=host, port=port, db=db,
                    decode_responses=True,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
                self.redis_client.ping()
                attempt = 0  # reset counter on successful connect
                logger.info(f"[RedisPubSub] Connected to Redis at {host}:{port}")

                self.pubsub = self.redis_client.pubsub()
                self.pubsub.subscribe(channel)
                logger.info(f"[RedisPubSub] Subscribed to channel: {channel}")

                # ── Listen ───────────────────────────────────────────────
                for message in self.pubsub.listen():
                    if not self.running:
                        break

                    if message['type'] == 'message':
                        try:
                            if message['type'] == 'message':
                                raw = message['data']
                                data, metadata = parse_redis_message(raw)
                                msg_type = metadata.get('type', 'DATA')
                                self._enqueue_message({
                                    'data': data, 'metadata': metadata, 'type': msg_type,
                                })
                                # flush immediato ad ogni messaggio — non aspettare il timer
                                self._flush_batch()
                        except json.JSONDecodeError as e:
                            logger.error(f"[RedisPubSub] Invalid JSON: {e}")
                            self.error_occurred.emit(f"Invalid JSON: {e}")
                        except Exception as e:
                            logger.error(f"[RedisPubSub] Event error: {e}\n{traceback.format_exc()}")
                            self.error_occurred.emit(f"Event processing error: {e}")

                    now = time.time()
                    if now - self.last_emit_time >= self.batch_interval:
                        self._flush_batch()

            except (RedisConnectionError, RedisTimeoutError, OSError) as e:
                # ── Reconnect ────────────────────────────────────────────
                attempt += 1
                if max_retries and attempt > max_retries:
                    logger.error(f"[RedisPubSub] Max retries ({max_retries}) exceeded. Stopping.")
                    self.error_occurred.emit(f"Redis max retries exceeded: {e}")
                    break

                delay = min(retry_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"[RedisPubSub] Connection lost (attempt {attempt}): {e} — "
                    f"reconnecting in {delay:.1f}s"
                )
                self.error_occurred.emit(f"Redis reconnecting in {delay:.0f}s…")
                self._cleanup_redis()
                time.sleep(delay)

            except Exception as e:
                logger.error(f"[RedisPubSub] Fatal error: {e}\n{traceback.format_exc()}")
                self.error_occurred.emit(f"Redis error: {e}")
                break

        self._flush_batch()
        self._cleanup_redis()
        logger.info("[RedisPubSub] Thread stopped")

    def _process_single_message(self, message: dict):
        """
        Processa singolo messaggio normalizzato.

        AGGIORNAMENTO: Lavora con formato normalizzato da parser.
        """
        msg_type = message.get('type', 'DATA')
        data = message.get('data')
        metadata = message.get('metadata', {})

        if msg_type in ('data', 'DATA'):
            try:
                df = self._to_dataframe(data)
                if df is None:
                    logger.warning("[RedisPubSub] _to_dataframe returned None, skipping emit")
                    return
                logger.debug(f"[RedisPubSub] Emitting data_updated: {len(df)} rows, cols={list(df.columns)}")
                self.data_updated.emit(df)
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"[RedisPubSub] Data conversion error: {e}\n{error_detail}")
                self.error_occurred.emit(f"Data conversion error: {e}")

        elif msg_type in ('status', 'flow_detected', 'command', 'config', 'error'):
            logger.debug(f"[RedisPubSub] Status/event message type={msg_type}")
            status_data = {
                'type': msg_type,
                'data': data
            }
            if metadata:
                status_data['metadata'] = metadata
            self.status_updated.emit(status_data)

    def _process_message(self, msg):
        """
        Gestisce sia un singolo messaggio che una LISTA di messaggi.
        Per 'data': concatena tutti i DataFrame e invia un solo update.

        AGGIORNAMENTO: Lavora con messaggi normalizzati.
        """
        # Caso 1: Lista di messaggi -> process batch
        if isinstance(msg, list):
            data_frames = []

            for single in msg:
                msg_type = single.get("type", "DATA")

                if msg_type in ("data", "DATA"):
                    data = single.get("data")
                    if data is not None:
                        try:
                            df = self._to_dataframe(data)
                            data_frames.append(df)
                        except Exception as e:
                            self.error_occurred.emit(f"Data conversion error: {str(e)}")
                else:
                    # Altri tipi: processa singolarmente
                    self._process_single_message(single)

            # Concatena DataFrame se presenti
            if data_frames:
                try:
                    df_concat = safe_concat(data_frames, ignore_index=True)
                    self.data_updated.emit(df_concat)
                except Exception as e:
                    self.error_occurred.emit(f"Batch dataframe concat error: {str(e)}")

            return

        # Caso 2: Singolo messaggio
        self._process_single_message(msg)

    def _enqueue_message(self, msg):
        """Accoda il messaggio e invia batch se necessario."""
        now = time.time()
        self.batch_queue.append(msg)

        # flush se superiamo dimensione massima
        if self.batch_max_size and len(self.batch_queue) >= self.batch_max_size:
            self._flush_batch()
            return

        # flush se superiamo intervallo
        if now - self.last_emit_time >= self.batch_interval:
            self._flush_batch()

    def _flush_batch(self):
        """Invia alla gui un batch di messaggi in una sola soluzione."""
        if not self.batch_queue:
            return

        batch = list(self.batch_queue)
        self.batch_queue.clear()
        self.last_emit_time = time.time()

        # Emetti un unico evento verso Qt
        try:
            self._process_message(batch)
        except Exception as e:
            self.error_occurred.emit(f"Batch processing error: {str(e)}")

    def _cleanup_redis(self):
        """Cleanup RedisPublisher connections"""
        if self.pubsub:
            try:
                self.pubsub.unsubscribe()
                self.pubsub.close()
            except:
                pass

        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass

    def stop(self):
        """Stop the thread and cleanup"""
        self.running = False
        self._cleanup_redis()
        self.wait(2000)


# ============================================================================
# CLASSE 4: RabbitMQ Pub/Sub Thread
# ============================================================================
class RabbitPubSubThread(BaseDashboardThread):
    """
    Worker thread per RabbitMQ mode.
    Sottoscrive a un exchange fanout e riceve messaggi in real-time.

    Equivalente di RedisPubSubThread ma per RabbitMQ:
    - Usa exchange fanout (broadcast a tutti i consumer registrati).
    - Crea una coda esclusiva temporanea per ogni istanza dashboard.
    - Stesso meccanismo di batching di RedisPubSubThread.
    - Stesso formato messaggi JSON (parse_redis_message riutilizzato).

    Configurazione attesa (rabbit_config dict):
        host         : str  - host RabbitMQ (default 'rabbitmq.af.tst')
        port         : int  - porta (default 5672)
        user         : str  - utente (default 'mqclient')
        password     : str  - password (default 'Mqclient-00')
        virtual_host : str  - vhost (default '/')
        exchange     : str  - nome exchange da sottoscrivere (default 'trades_df')
    """

    def __init__(self, rabbit_config: Dict):
        super().__init__()
        self.batch_queue = deque()
        self.last_emit_time = 0
        self.batch_interval = rabbit_config.get('batch_interval', 0.5)  # 500ms
        self.batch_max_size = None

        self.rabbit_config = rabbit_config
        self._connection = None
        self._channel = None
        self._queue_name = None

    def run(self):
        """Main loop - RabbitMQ consumer con batching."""
        try:
            import pika
            from pika.exceptions import AMQPConnectionError
        except ImportError:
            self.error_occurred.emit(
                "pika non installato. Eseguire: pip install pika"
            )
            return

        self.running = True
        exchange = self.rabbit_config.get('exchange', 'trades_df')
        host = self.rabbit_config.get('host', 'rabbitmq.af.tst')
        port = self.rabbit_config.get('port', 5672)
        logger.info(f"[RabbitPubSub] Thread started — host={host}:{port} exchange={exchange}")

        try:
            credentials = pika.PlainCredentials(
                username=self.rabbit_config.get('user', 'mqclient'),
                password=self.rabbit_config.get('password', 'Mqclient-00'),
            )
            params = pika.ConnectionParameters(
                host=self.rabbit_config.get('host', 'rabbitmq.af.tst'),
                port=self.rabbit_config.get('port', 5672),
                virtual_host=self.rabbit_config.get('virtual_host', '/'),
                credentials=credentials,
                heartbeat=60,
                blocked_connection_timeout=30,
            )

            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()

            exchange = self.rabbit_config.get('exchange', 'trades_df')
            logger.info(f"[RabbitPubSub] Connected to RabbitMQ at {host}:{port}")

            # Dichiara exchange fanout (idempotente)
            self._channel.exchange_declare(
                exchange=exchange,
                exchange_type='fanout',
                durable=True,
            )

            # Coda esclusiva temporanea (auto-delete alla disconnessione)
            result = self._channel.queue_declare(
                queue='', exclusive=True, auto_delete=True
            )
            self._queue_name = result.method.queue

            # Bind coda -> exchange
            self._channel.queue_bind(
                exchange=exchange, queue=self._queue_name
            )

            logger.info(
                f"[RabbitPubSub] Subscribed to exchange: {exchange}, "
                f"queue: {self._queue_name}"
            )

            def _on_message(ch, method, properties, body):
                """Callback invocata per ogni messaggio ricevuto."""
                if not self.running:
                    ch.stop_consuming()
                    return
                try:
                    json_str = body.decode('utf-8')
                    logger.debug(f"[RabbitPubSub] Raw message received, len={len(json_str)}")
                    data, metadata = parse_redis_message(json_str)
                    msg_type = metadata.get('type', 'DATA')
                    logger.debug(f"[RabbitPubSub] Parsed message type={msg_type}")
                    normalized_msg = {
                        'data': data,
                        'metadata': metadata,
                        'type': msg_type,
                    }
                    self._enqueue_message(normalized_msg)
                except json.JSONDecodeError as e:
                    logger.error(f"[RabbitPubSub] Invalid JSON: {e}")
                    self.error_occurred.emit(f"Invalid JSON from RabbitMQ: {e}")
                except Exception as e:
                    error_detail = traceback.format_exc()
                    logger.error(f"[RabbitPubSub] Message processing error: {e}\n{error_detail}")
                    self.error_occurred.emit(f"RabbitMQ message processing error: {e}")

                # Flush batch se l'intervallo è scaduto
                now = time.time()
                if now - self.last_emit_time >= self.batch_interval:
                    self._flush_batch()

            self._channel.basic_consume(
                queue=self._queue_name,
                on_message_callback=_on_message,
                auto_ack=True,
            )

            # Blocking loop - gestisce heartbeat e dispatch messaggi
            self._channel.start_consuming()

        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"[RabbitPubSub] Fatal error: {e}\n{error_detail}")
            self.error_occurred.emit(f"RabbitMQ error: {e}")

        finally:
            self._flush_batch()
            self._cleanup_rabbit()
            logger.info("[RabbitPubSub] Thread stopped")

    # ------------------------------------------------------------------
    # Batching (speculare a RedisPubSubThread)
    # ------------------------------------------------------------------

    def _enqueue_message(self, msg):
        """Accoda il messaggio e invia batch se necessario."""
        now = time.time()
        self.batch_queue.append(msg)

        if self.batch_max_size and len(self.batch_queue) >= self.batch_max_size:
            self._flush_batch()
            return

        if now - self.last_emit_time >= self.batch_interval:
            self._flush_batch()

    def _flush_batch(self):
        """Invia alla GUI un batch di messaggi in una sola soluzione."""
        if not self.batch_queue:
            return

        batch = list(self.batch_queue)
        self.batch_queue.clear()
        self.last_emit_time = time.time()

        try:
            self._process_message(batch)
        except Exception as e:
            self.error_occurred.emit(f"Batch processing error: {str(e)}")

    # ------------------------------------------------------------------
    # Processing (identico a RedisPubSubThread)
    # ------------------------------------------------------------------

    def _process_single_message(self, message: dict):
        """Processa singolo messaggio normalizzato."""
        msg_type = message.get('type', 'DATA')
        data = message.get('data')
        metadata = message.get('metadata', {})

        if msg_type in ('data', 'DATA'):
            try:
                df = self._to_dataframe(data)
                if df is None:
                    logger.warning("[RabbitPubSub] _to_dataframe returned None, skipping emit")
                    return
                logger.debug(f"[RabbitPubSub] Emitting data_updated: {len(df)} rows, cols={list(df.columns)}")
                self.data_updated.emit(df)
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"[RabbitPubSub] Data conversion error: {e}\n{error_detail}")
                self.error_occurred.emit(f"Data conversion error: {e}")

        elif msg_type in ('status', 'flow_detected', 'command', 'config', 'error'):
            logger.debug(f"[RabbitPubSub] Status/event message type={msg_type}")
            status_data = {'type': msg_type, 'data': data}
            if metadata:
                status_data['metadata'] = metadata
            self.status_updated.emit(status_data)

    def _process_message(self, msg):
        """Gestisce lista o singolo messaggio normalizzato."""
        if isinstance(msg, list):
            data_frames = []
            for single in msg:
                msg_type = single.get("type", "DATA")
                if msg_type in ("data", "DATA"):
                    data = single.get("data")
                    if data is not None:
                        try:
                            df = self._to_dataframe(data)
                            data_frames.append(df)
                        except Exception as e:
                            self.error_occurred.emit(
                                f"Data conversion error: {str(e)}"
                            )
                else:
                    self._process_single_message(single)

            if data_frames:
                try:
                    df_concat = safe_concat(data_frames, ignore_index=True)
                    self.data_updated.emit(df_concat)
                except Exception as e:
                    self.error_occurred.emit(
                        f"Batch dataframe concat error: {str(e)}"
                    )
            return

        self._process_single_message(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_rabbit(self):
        """Cleanup connessioni RabbitMQ."""
        if self._channel:
            try:
                if self._channel.is_open:
                    self._channel.close()
            except Exception:
                pass

        if self._connection:
            try:
                if self._connection.is_open:
                    self._connection.close()
            except Exception:
                pass

    def stop(self):
        """Stop del thread e cleanup."""
        self.running = False
        # Interrompe il blocking start_consuming() in modo thread-safe
        if self._connection and self._connection.is_open:
            try:
                self._connection.add_callback_threadsafe(
                    self._channel.stop_consuming
                )
            except Exception:
                pass
        self.wait(3000)