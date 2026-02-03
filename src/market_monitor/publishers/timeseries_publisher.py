"""
TimeSeriesPublisher - High-performance Redis TimeSeries publisher for MarketMonitor.

Estende RedisPublisher per supportare Redis TimeSeries con:
- Batch/pipeline publishing per alte performance
- Regole di persistenza configurabili (retention, compaction, duplicate policy)
- Creazione automatica delle TimeSeries con labels
- Change detection per evitare duplicati
- Multi-field publishing (nav, mid, bid, ask, etc.)
- Integrazione con DataNormalizer esistente

Uso:
    from market_monitor.publishers.timeseries_publisher import TimeSeriesPublisher, RetentionPolicy

    # Crea publisher
    publisher = TimeSeriesPublisher(redis_host="localhost", redis_port=6380)

    # Pubblica singolo valore
    publisher.ts_add("IT0001234567", "nav", 100.50)

    # Pubblica multipli field per stesso ISIN
    publisher.ts_add_multi("IT0001234567", {"nav": 100.50, "mid": 100.45, "bid": 100.40})

    # Batch publishing (molto più performante)
    with publisher.ts_batch() as batch:
        batch.add("IT0001", "nav", 100.0)
        batch.add("IT0002", "nav", 50.0)

    # Configura retention personalizzata
    publisher.ts_set_retention("IT0001234567", "nav", retention_ms=86400000)
"""

import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import threading

import redis

from market_monitor.publishers.redis_publisher import RedisPublisher

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class DuplicatePolicy(Enum):
    """Policy per gestione timestamp duplicati."""
    BLOCK = "BLOCK"       # Errore se duplicato
    FIRST = "FIRST"       # Mantieni primo valore
    LAST = "LAST"         # Mantieni ultimo valore (default)
    MIN = "MIN"           # Mantieni valore minimo
    MAX = "MAX"           # Mantieni valore massimo
    SUM = "SUM"           # Somma i valori


class AggregationType(Enum):
    """Tipi di aggregazione per compaction rules."""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STD_P = "std.p"
    STD_S = "std.s"
    VAR_P = "var.p"
    VAR_S = "var.s"
    TWA = "twa"


@dataclass
class CompactionRule:
    """Regola di compaction per downsampling automatico."""
    aggregation: AggregationType
    bucket_duration_ms: int      # Durata bucket in ms (es: 3600000 = 1 ora)
    destination_key_suffix: str  # Suffisso per chiave destinazione (es: "_hourly")
    retention_ms: Optional[int] = None  # Retention per la serie compattata

    def __post_init__(self):
        if isinstance(self.aggregation, str):
            self.aggregation = AggregationType(self.aggregation)


@dataclass
class RetentionPolicy:
    """Policy di retention per una TimeSeries."""
    retention_ms: int = 86400000  # Default: 24 ore
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.LAST
    chunk_size: int = 4096  # Dimensione chunk in bytes
    compaction_rules: List[CompactionRule] = field(default_factory=list)

    # Preset comuni
    @classmethod
    def intraday(cls) -> 'RetentionPolicy':
        """24 ore, nessuna compaction."""
        return cls(retention_ms=86400000)

    @classmethod
    def weekly(cls) -> 'RetentionPolicy':
        """7 giorni con compaction oraria."""
        return cls(
            retention_ms=7 * 86400000,
            compaction_rules=[
                CompactionRule(AggregationType.LAST, 3600000, "_hourly", 30 * 86400000)
            ]
        )

    @classmethod
    def monthly(cls) -> 'RetentionPolicy':
        """30 giorni con compaction oraria e giornaliera."""
        return cls(
            retention_ms=30 * 86400000,
            compaction_rules=[
                CompactionRule(AggregationType.LAST, 3600000, "_hourly", 90 * 86400000),
                CompactionRule(AggregationType.LAST, 86400000, "_daily", 365 * 86400000)
            ]
        )

    @classmethod
    def yearly(cls) -> 'RetentionPolicy':
        """365 giorni con compaction completa."""
        return cls(
            retention_ms=365 * 86400000,
            compaction_rules=[
                CompactionRule(AggregationType.LAST, 3600000, "_hourly", 90 * 86400000),
                CompactionRule(AggregationType.LAST, 86400000, "_daily", 5 * 365 * 86400000),
            ]
        )

    @classmethod
    def forever(cls) -> 'RetentionPolicy':
        """Nessun limite di retention (0 = infinito)."""
        return cls(
            retention_ms=0,
            compaction_rules=[
                CompactionRule(AggregationType.LAST, 86400000, "_daily", 0)
            ]
        )


# ============================================================================
# Batch Context Manager
# ============================================================================

class TSBatchContext:
    """Context manager per batch publishing di TimeSeries."""

    def __init__(self, publisher: 'TimeSeriesPublisher'):
        self.publisher = publisher
        self.operations: List[Tuple[str, int, float]] = []  # (key, timestamp_ms, value)

    def add(
        self,
        identifier: str,
        field: str,
        value: float,
        timestamp: Optional[Union[datetime, int, float]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """Aggiunge operazione al batch."""
        key = self.publisher._build_ts_key(identifier, field)
        ts_ms = self.publisher._normalize_timestamp(timestamp)

        # Assicura che la TS esista
        self.publisher._ensure_timeseries_exists(key, identifier, field, labels)

        self.operations.append((key, ts_ms, value))

    def add_multi(
        self,
        identifier: str,
        values: Dict[str, float],
        timestamp: Optional[Union[datetime, int, float]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        """Aggiunge multipli field per stesso identifier."""
        ts_ms = self.publisher._normalize_timestamp(timestamp)

        for field_name, value in values.items():
            key = self.publisher._build_ts_key(identifier, field_name)
            self.publisher._ensure_timeseries_exists(key, identifier, field_name, labels)
            self.operations.append((key, ts_ms, value))

    def __len__(self):
        return len(self.operations)


# ============================================================================
# TimeSeriesPublisher Class
# ============================================================================

class TimeSeriesPublisher(RedisPublisher):
    """
    Publisher ad alte performance per Redis TimeSeries.

    Estende RedisPublisher aggiungendo:
    - Metodi ts_* per operazioni TimeSeries
    - Pipeline/batch support per ridurre round-trip
    - Auto-creazione TimeSeries con labels
    - Regole di persistenza configurabili
    - Thread-safe
    - Change detection opzionale

    Mantiene compatibilità con RedisPublisher esistente (Pub/Sub, SET, etc.)
    """

    # Key format: ts:{identifier}:{field}
    TS_KEY_PREFIX = "ts"

    def __init__(
        self,
        *args,
        redis_host: str = 'localhost',
        redis_port: int = 6380,  # Default Redis Stack port
        redis_db: int = 0,
        default_policy: Optional[RetentionPolicy] = None,
        auto_create: bool = True,
        skip_duplicates: bool = True,
        **kwargs
    ):
        """
        Inizializza il publisher con supporto TimeSeries.

        Args:
            redis_host: Redis host
            redis_port: Redis port (default 6380 per Redis Stack)
            redis_db: Redis database number
            default_policy: Policy di retention di default
            auto_create: Crea automaticamente TimeSeries se non esistono
            skip_duplicates: Salta valori duplicati (stesso timestamp e valore)
            **kwargs: Altri parametri per RedisPublisher parent
        """
        super().__init__(
            *args,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            **kwargs
        )

        self.default_policy = default_policy or RetentionPolicy.intraday()
        self.auto_create = auto_create
        self.skip_duplicates = skip_duplicates

        # Cache per TimeSeries esistenti
        self._existing_ts_keys: set = set()
        self._existing_ts_keys_lock = threading.Lock()

        # Cache per change detection TimeSeries
        self._ts_last_values: Dict[str, Tuple[int, float]] = {}  # key -> (timestamp, value)

        # Statistiche
        self.ts_stats = {
            "total_published": 0,
            "duplicates_skipped": 0,
            "timeseries_created": 0,
            "batch_operations": 0,
            "errors": 0
        }

        logger.info(f"TimeSeriesPublisher initialized: {redis_host}:{redis_port}")

    def _build_ts_key(self, identifier: str, field: str) -> str:
        """Costruisce chiave TimeSeries: ts:{identifier}:{field}"""
        return f"{self.TS_KEY_PREFIX}:{identifier}:{field.lower()}"

    def _normalize_timestamp(self, timestamp: Optional[Union[datetime, int, float]] = None) -> int:
        """Normalizza timestamp a millisecondi Unix."""
        if timestamp is None:
            return int(time.time() * 1000)
        elif isinstance(timestamp, datetime):
            return int(timestamp.timestamp() * 1000)
        elif isinstance(timestamp, float):
            # Assume seconds se < 1e12, altrimenti millisecondi
            if timestamp < 1e12:
                return int(timestamp * 1000)
            return int(timestamp)
        else:
            return int(timestamp)

    def _ensure_timeseries_exists(
        self,
        key: str,
        identifier: str,
        field: str,
        labels: Optional[Dict[str, str]] = None,
        policy: Optional[RetentionPolicy] = None
    ):
        """Crea TimeSeries se non esiste (con caching)."""
        if not self.auto_create:
            return

        with self._existing_ts_keys_lock:
            if key in self._existing_ts_keys:
                return

        policy = policy or self.default_policy

        # Labels di default
        default_labels = {
            "identifier": identifier,
            "field": field
        }
        if labels:
            default_labels.update(labels)

        try:
            self.redis_client.execute_command(
                "TS.CREATE", key,
                "RETENTION", policy.retention_ms,
                "DUPLICATE_POLICY", policy.duplicate_policy.value.lower(),
                "CHUNK_SIZE", policy.chunk_size,
                "LABELS", *[item for pair in default_labels.items() for item in pair]
            )

            # Crea compaction rules
            for rule in policy.compaction_rules:
                dest_key = f"{key}{rule.destination_key_suffix}"

                # Crea serie destinazione
                try:
                    dest_labels = {**default_labels, "aggregation": rule.aggregation.value}
                    self.redis_client.execute_command(
                        "TS.CREATE", dest_key,
                        "RETENTION", rule.retention_ms or 0,
                        "LABELS", *[item for pair in dest_labels.items() for item in pair]
                    )
                except redis.ResponseError:
                    pass  # Già esiste

                # Crea regola
                try:
                    self.redis_client.execute_command(
                        "TS.CREATERULE", key, dest_key,
                        "AGGREGATION", rule.aggregation.value, rule.bucket_duration_ms
                    )
                except redis.ResponseError:
                    pass  # Regola già esiste

            with self._existing_ts_keys_lock:
                self._existing_ts_keys.add(key)

            self.ts_stats["timeseries_created"] += 1
            logger.debug(f"Created TimeSeries: {key}")

        except redis.ResponseError as e:
            if "already exists" in str(e).lower():
                with self._existing_ts_keys_lock:
                    self._existing_ts_keys.add(key)
            else:
                raise

    def _should_publish_ts(self, key: str, timestamp_ms: int, value: float) -> bool:
        """Verifica se il valore deve essere pubblicato (change detection)."""
        if not self.skip_duplicates:
            return True

        last = self._ts_last_values.get(key)
        if last is None:
            return True

        last_ts, last_val = last
        # Pubblica se timestamp diverso o valore diverso
        return timestamp_ms != last_ts or value != last_val

    # ========================================================================
    # Public TimeSeries Methods
    # ========================================================================

    def ts_add(
        self,
        identifier: str,
        field: str,
        value: float,
        timestamp: Optional[Union[datetime, int, float]] = None,
        labels: Optional[Dict[str, str]] = None,
        policy: Optional[RetentionPolicy] = None
    ) -> bool:
        """
        Pubblica singolo valore su TimeSeries.

        Args:
            identifier: Identificativo (es: ISIN)
            field: Campo (es: nav, mid, bid, ask)
            value: Valore numerico
            timestamp: Timestamp (default: now)
            labels: Labels aggiuntive
            policy: Policy di retention (default: usa default_policy)

        Returns:
            True se pubblicato, False se skippato
        """
        key = self._build_ts_key(identifier, field)
        ts_ms = self._normalize_timestamp(timestamp)

        # Change detection
        if not self._should_publish_ts(key, ts_ms, value):
            self.ts_stats["duplicates_skipped"] += 1
            return False

        try:
            self._ensure_timeseries_exists(key, identifier, field, labels, policy)

            self.redis_client.execute_command("TS.ADD", key, ts_ms, value)

            # Aggiorna cache
            self._ts_last_values[key] = (ts_ms, value)
            self.ts_stats["total_published"] += 1

            return True

        except redis.ResponseError as e:
            if "DUPLICATE" in str(e).upper():
                self.ts_stats["duplicates_skipped"] += 1
                return False
            self.ts_stats["errors"] += 1
            logger.error(f"Error publishing {key}: {e}")
            raise

    def ts_add_multi(
        self,
        identifier: str,
        values: Dict[str, float],
        timestamp: Optional[Union[datetime, int, float]] = None,
        labels: Optional[Dict[str, str]] = None,
        policy: Optional[RetentionPolicy] = None
    ) -> int:
        """
        Pubblica multipli field per stesso identifier.

        Args:
            identifier: Identificativo (es: ISIN)
            values: Dict {field: value} (es: {"nav": 100.0, "mid": 99.95})
            timestamp: Timestamp comune
            labels: Labels aggiuntive
            policy: Policy di retention

        Returns:
            Numero di valori pubblicati
        """
        ts_ms = self._normalize_timestamp(timestamp)
        published = 0

        # Usa pipeline per performance
        pipe = self.redis_client.pipeline()

        for field_name, value in values.items():
            key = self._build_ts_key(identifier, field_name)

            if not self._should_publish_ts(key, ts_ms, value):
                self.ts_stats["duplicates_skipped"] += 1
                continue

            self._ensure_timeseries_exists(key, identifier, field_name, labels, policy)
            pipe.execute_command("TS.ADD", key, ts_ms, value)

            # Aggiorna cache
            self._ts_last_values[key] = (ts_ms, value)
            published += 1

        if published > 0:
            try:
                pipe.execute()
                self.ts_stats["total_published"] += published
            except redis.ResponseError as e:
                self.ts_stats["errors"] += 1
                logger.error(f"Error in ts_add_multi: {e}")
                raise

        return published

    @contextmanager
    def ts_batch(self):
        """
        Context manager per batch publishing.

        Uso:
            with publisher.ts_batch() as batch:
                batch.add("ISIN1", "nav", 100.0)
                batch.add("ISIN2", "nav", 50.0)
                # ... molte operazioni
            # Commit automatico all'uscita
        """
        ctx = TSBatchContext(self)
        try:
            yield ctx
        finally:
            if ctx.operations:
                self._execute_ts_batch(ctx.operations)

    def _execute_ts_batch(self, operations: List[Tuple[str, int, float]]):
        """Esegue batch di operazioni con pipeline."""
        if not operations:
            return

        pipe = self.redis_client.pipeline()

        for key, ts_ms, value in operations:
            pipe.execute_command("TS.ADD", key, ts_ms, value)
            self._ts_last_values[key] = (ts_ms, value)

        try:
            results = pipe.execute()
            successful = sum(1 for r in results if not isinstance(r, Exception))
            self.ts_stats["total_published"] += successful
            self.ts_stats["batch_operations"] += 1
            logger.debug(f"Batch published {successful}/{len(operations)} values")
        except redis.ResponseError as e:
            self.ts_stats["errors"] += 1
            logger.error(f"Batch error: {e}")
            raise

    def ts_madd(
        self,
        data: List[Tuple[str, str, float, Optional[Union[datetime, int, float]]]],
        labels: Optional[Dict[str, str]] = None,
        policy: Optional[RetentionPolicy] = None
    ) -> int:
        """
        Multi-add: pubblica lista di (identifier, field, value, timestamp).

        Uso più performante per grandi quantità di dati.

        Args:
            data: Lista di tuple (identifier, field, value, timestamp)
            labels: Labels comuni
            policy: Policy comune

        Returns:
            Numero di valori pubblicati
        """
        if not data:
            return 0

        pipe = self.redis_client.pipeline()
        count = 0

        for item in data:
            if len(item) == 3:
                identifier, field_name, value = item
                timestamp = None
            else:
                identifier, field_name, value, timestamp = item

            key = self._build_ts_key(identifier, field_name)
            ts_ms = self._normalize_timestamp(timestamp)

            if not self._should_publish_ts(key, ts_ms, value):
                self.ts_stats["duplicates_skipped"] += 1
                continue

            self._ensure_timeseries_exists(key, identifier, field_name, labels, policy)
            pipe.execute_command("TS.ADD", key, ts_ms, value)
            self._ts_last_values[key] = (ts_ms, value)
            count += 1

        if count > 0:
            try:
                pipe.execute()
                self.ts_stats["total_published"] += count
            except redis.ResponseError as e:
                self.ts_stats["errors"] += 1
                raise

        return count

    # ========================================================================
    # TimeSeries Management Methods
    # ========================================================================

    def ts_create(
        self,
        identifier: str,
        field: str,
        labels: Optional[Dict[str, str]] = None,
        policy: Optional[RetentionPolicy] = None
    ) -> str:
        """Crea esplicitamente una TimeSeries."""
        key = self._build_ts_key(identifier, field)
        self._ensure_timeseries_exists(key, identifier, field, labels, policy)
        return key

    def ts_set_retention(
        self,
        identifier: str,
        field: str,
        retention_ms: int
    ):
        """Modifica retention di una TimeSeries esistente."""
        key = self._build_ts_key(identifier, field)
        self.redis_client.execute_command("TS.ALTER", key, "RETENTION", retention_ms)
        logger.info(f"Updated retention for {key}: {retention_ms}ms")

    def ts_add_compaction_rule(
        self,
        identifier: str,
        field: str,
        rule: CompactionRule
    ):
        """Aggiunge regola di compaction a TimeSeries esistente."""
        source_key = self._build_ts_key(identifier, field)
        dest_key = f"{source_key}{rule.destination_key_suffix}"

        # Crea destinazione
        try:
            info = self.redis_client.execute_command("TS.INFO", source_key)
            # Parse labels from info
            labels = {"aggregation": rule.aggregation.value}

            self.redis_client.execute_command(
                "TS.CREATE", dest_key,
                "RETENTION", rule.retention_ms or 0,
                "LABELS", *[item for pair in labels.items() for item in pair]
            )
        except redis.ResponseError:
            pass

        # Crea regola
        self.redis_client.execute_command(
            "TS.CREATERULE", source_key, dest_key,
            "AGGREGATION", rule.aggregation.value, rule.bucket_duration_ms
        )
        logger.info(f"Added compaction rule: {source_key} -> {dest_key}")

    def ts_delete(self, identifier: str, field: str):
        """Elimina una TimeSeries."""
        key = self._build_ts_key(identifier, field)
        self.redis_client.delete(key)
        with self._existing_ts_keys_lock:
            self._existing_ts_keys.discard(key)
        logger.info(f"Deleted TimeSeries: {key}")

    def ts_get_info(self, identifier: str, field: str) -> dict:
        """Ottiene informazioni su una TimeSeries."""
        key = self._build_ts_key(identifier, field)
        info = self.redis_client.execute_command("TS.INFO", key)

        # Parse info response (list of key-value pairs)
        info_dict = {}
        for i in range(0, len(info), 2):
            info_dict[info[i]] = info[i + 1]

        return {
            "total_samples": info_dict.get("totalSamples", 0),
            "memory_usage": info_dict.get("memoryUsage", 0),
            "first_timestamp": info_dict.get("firstTimestamp", 0),
            "last_timestamp": info_dict.get("lastTimestamp", 0),
            "retention_msecs": info_dict.get("retentionTime", 0),
            "labels": info_dict.get("labels", {})
        }

    def ts_get_last(self, identifier: str, field: str) -> Optional[Tuple[int, float]]:
        """Ottiene ultimo valore di una TimeSeries."""
        key = self._build_ts_key(identifier, field)
        try:
            result = self.redis_client.execute_command("TS.GET", key)
            return (result[0], result[1]) if result else None
        except redis.ResponseError:
            return None

    def ts_list(self, identifier: Optional[str] = None) -> List[str]:
        """Lista TimeSeries per identifier (o tutte se None)."""
        pattern = f"{self.TS_KEY_PREFIX}:{identifier or '*'}:*"
        return list(self.redis_client.scan_iter(match=pattern, _type="TSDB-TYPE"))

    def ts_clear_cache(self):
        """Pulisce cache interne TimeSeries."""
        self._ts_last_values.clear()
        with self._existing_ts_keys_lock:
            self._existing_ts_keys.clear()
        logger.debug("TimeSeries cache cleared")

    def ts_get_stats(self) -> dict:
        """Ritorna statistiche TimeSeries publisher."""
        return dict(self.ts_stats)


# ============================================================================
# Convenience function for quick usage
# ============================================================================

def get_timeseries_publisher(
    host: str = "localhost",
    port: int = 6380,
    policy: Optional[RetentionPolicy] = None
) -> TimeSeriesPublisher:
    """
    Factory function per creare un TimeSeriesPublisher.

    Args:
        host: Redis host
        port: Redis port (default 6380 per Redis Stack)
        policy: Policy di retention (default: intraday)

    Returns:
        TimeSeriesPublisher configurato
    """
    return TimeSeriesPublisher(
        redis_host=host,
        redis_port=port,
        default_policy=policy or RetentionPolicy.intraday()
    )
