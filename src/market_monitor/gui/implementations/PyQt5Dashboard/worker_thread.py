"""
Worker threads per TradeDashboard - VERSIONE AGGIORNATA.
Gestiscono polling dati da Queue o RedisPublisher Pub/Sub.

AGGIORNAMENTO: Supporto completo per formato flat RTD-compatible.
"""
import json
from collections import deque
from unittest.util import safe_repr

from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
from typing import Dict, Any, Tuple
from queue import Empty
import time

from market_monitor.gui.implementations.PyQt5Dashboard.common import safe_concat


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
        print(f"Errore parsing JSON: {e}")
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

    def _to_dataframe(self, data):
        """Convert data to DataFrame"""
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, str):
            try:
                df = pd.DataFrame.from_records(json.loads(data))
            except Exception as e:
                print("dump df as records", e)
                return None
        else:
            return None
        
        # CONVERSIONE TIMESTAMP: Converti colonne timestamp in datetime pandas
        timestamp_cols = ['timestamp', 'time', 'datetime', 'last_update']
        
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {e}")

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
        print(f"[QueuePolling] Thread started")

        while self.running:
            try:
                # Blocking get con timeout dalla queue
                message = self.queue.get(timeout=0.05)

                # ✅ NUOVO: Processa con parser universale
                self._process_message_universal(message)

            except Empty:
                # Nessun messaggio, continua
                continue

            except Exception as e:
                self.error_occurred.emit(f"Queue polling error: {str(e)}")
                time.sleep(0.1)

        print("[QueuePolling] Thread stopped")

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
                print(f"[QueuePolling] Unknown message type: {msg_type}, treating as data")
                self._handle_data(data, metadata)

        except Exception as e:
            self.error_occurred.emit(f"Message processing error: {str(e)}")

    def _handle_data(self, data: Any, metadata: Dict):
        """Gestisce messaggi di tipo data"""
        try:
            df = self._to_dataframe(data)
            self.data_updated.emit(df)
        except Exception as e:
            self.error_occurred.emit(f"Data conversion error: {str(e)}")

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
        self.batch_interval = 0.5  # 500ms
        self.batch_max_size = None

        self.redis_config = redis_config
        self.redis_client = None
        self.pubsub = None

    def run(self):
        """Main loop - RedisPublisher Pub/Sub listener with batching."""
        try:
            import redis
        except ImportError:
            self.error_occurred.emit(
                "RedisPublisher not installed. Run: pip install redis"
            )
            return

        self.running = True
        print(f"[RedisPubSub] Thread started")

        try:
            # Connetti RedisPublisher
            self.redis_client = redis.Redis(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 0),
                decode_responses=True
            )

            # Test connessione
            self.redis_client.ping()

            # Sottoscrivi canale
            channel = self.redis_config.get('channel', 'trades')
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe(channel)

            print(f"[RedisPubSub] Subscribed to channel: {channel}")

            # Listen messaggi
            for message in self.pubsub.listen():
                if not self.running:
                    break

                if message['type'] == 'message':
                    try:
                        # ✅ NUOVO: Parser universale
                        data, metadata = parse_redis_message(message['data'])

                        # Ricostruisci messaggio normalizzato
                        normalized_msg = {
                            'data': data,
                            'metadata': metadata,
                            'type': metadata.get('type', 'DATA')
                        }

                        # Enqueue per batching
                        self._enqueue_message(normalized_msg)

                    except json.JSONDecodeError as e:
                        self.error_occurred.emit(f"Invalid JSON: {str(e)}")
                    except Exception as e:
                        self.error_occurred.emit(f"Event processing error: {str(e)}")

                # Controllo flush batch
                now = time.time()
                if now - self.last_emit_time >= self.batch_interval:
                    self._flush_batch()

        except Exception as e:
            self.error_occurred.emit(f"RedisPublisher error: {str(e)}")

        finally:
            # flush finale dei messaggi residui
            self._flush_batch()
            self._cleanup_redis()

        print("[RedisPubSub] Thread stopped")

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
                self.data_updated.emit(df)
            except Exception as e:
                self.error_occurred.emit(f"Data conversion error: {str(e)}")

        elif msg_type in ('status', 'flow_detected', 'command', 'config', 'error'):
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
        # Caso 1: Lista di messaggi → process batch
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