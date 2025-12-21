import json
import logging
from typing import Any, Dict, Optional, Set, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import redis

from market_monitor.publishers.base import MessageType
from UserStrategy.utils.TradeManager.FlowDetector import Flow

logger = logging.getLogger(__name__)


# ============================================================================
# Data Normalization Layer
# ============================================================================

class DataNormalizer:
    """Normalizza diversi tipi di dati in strutture confrontabili."""

    @staticmethod
    def normalize(value: Any, **kwargs) -> Any:
        """
        Converte un valore in una forma normalizzata per il confronto.

        Args:
            value: Valore da normalizzare

        Returns:
            Valore normalizzato (dict, list, scalar) o None se tipo non supportato
        """
        if isinstance(value, pd.Series):
            return value.to_dict()
        elif isinstance(value, pd.DataFrame):
            return value.to_dict(orient=kwargs.get('orient', 'records'),
                                 index=kwargs.get('index', True))
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
            return value
        else:
            logger.warning(f"Tipo non supportato per normalizzazione: {type(value)}")
            return None


# ============================================================================
# Change Detection Layer
# ============================================================================

class ChangeDetector:
    """Rileva cambiamenti tra dati con supporto per confronti granulari."""

    def __init__(self):
        self.last_values: Dict[str, Any] = {}

    def has_changed(
            self,
            channel: str,
            value: Any,
            skip_if_unchanged: bool = False
    ) -> Tuple[bool, Any]:
        """
        Determina se un valore è cambiato rispetto all'ultima pubblicazione.

        Args:
            channel: Identificatore del canale
            value: Nuovo valore (già normalizzato)
            skip_if_unchanged: Se True, calcola diff granulare per dict

        Returns:
            Tuple (has_changed: bool, value_to_publish: Any)
            - has_changed: True se ci sono modifiche da pubblicare
            - value_to_publish: Valore completo o diff (solo chiavi cambiate per dict)
        """
        if not skip_if_unchanged:
            # Modalità standard: pubblica sempre il valore completo
            self.last_values[channel] = deepcopy(value)
            return True, value

        # Modalità skip: confronto granulare
        prev_value = self.last_values.get(channel)

        if prev_value is None:
            # Prima pubblicazione
            self.last_values[channel] = deepcopy(value)
            return True, value

        # Confronto basato sul tipo
        if isinstance(value, dict) and isinstance(prev_value, dict):
            return self._check_dict_changes(channel, value, prev_value)
        else:
            # Confronto semplice per altri tipi
            if value == prev_value:
                logger.debug(f"Nessuna modifica per '{channel}' (valore identico)")
                return False, None
            else:
                self.last_values[channel] = deepcopy(value)
                return True, value

    def _check_dict_changes(
            self,
            channel: str,
            new_dict: dict,
            prev_dict: dict
    ) -> Tuple[bool, Optional[dict]]:
        """
        Confronto granulare per dizionari: ritorna solo le chiavi modificate.

        Args:
            channel: Identificatore del canale
            new_dict: Nuovo dizionario
            prev_dict: Dizionario precedente

        Returns:
            Tuple (has_changes, diff_dict)
            - has_changes: True se ci sono modifiche
            - diff_dict: Dizionario con solo le chiavi modificate (o None se nessuna modifica)
        """
        diff = {}

        # Cerca chiavi nuove o modificate
        for key, val in new_dict.items():
            if key not in prev_dict or prev_dict[key] != val:
                diff[key] = val

        # Nota: le chiavi rimosse NON vengono incluse nel diff
        # Se serve gestirle, si può aggiungere un parametro e includere:
        # for key in prev_dict:
        #     if key not in new_dict:
        #         diff[key] = None  # o qualche valore speciale

        if not diff:
            logger.debug(f"Nessuna modifica per '{channel}' (dizionario identico)")
            return False, None

        # Aggiorna cache con i nuovi valori (merge, non replace)
        updated_dict = prev_dict.copy()
        updated_dict.update(new_dict)
        self.last_values[channel] = updated_dict

        logger.debug(f"Modifiche su '{channel}': {len(diff)} chiavi cambiate su {len(new_dict)} totali")
        return True, diff

    def clear_cache(self, channel: Optional[str] = None) -> None:
        """
        Pulisce la cache dei valori precedenti.

        Args:
            channel: Canale specifico da pulire (None = pulisce tutto)
        """
        if channel is None:
            self.last_values.clear()
            logger.debug("Cache completamente pulita")
        elif channel in self.last_values:
            del self.last_values[channel]
            logger.debug(f"Cache pulita per '{channel}'")


# ============================================================================
# Serialization Layer
# ============================================================================

class DataSerializer:
    """Serializza dati normalizzati in formato JSON per RedisPublisher."""

    @staticmethod
    def serialize(value: Any, flat_values: bool = False) -> Optional[str]:
        """
        Serializza un valore normalizzato in JSON.

        Args:
            value: Valore normalizzato da serializzare
            flat_values: Se True, usa formato piatto per valori scalari (RTD-compatible)

        Returns:
            Stringa JSON o None se non serializzabile
        """
        if value is None:
            return None

        try:
            if isinstance(value, str):
                # Stringhe già serializzate o plain text
                return value
            else:
                # Tutto il resto: serializza come JSON
                return json.dumps(value, default=str)
        except Exception as e:
            logger.error(f"Errore serializzazione: {e}")
            return None

    @staticmethod
    def serialize_with_metadata(
            data: Any,
            metadata: Optional[Dict[str, Any]] = None,
            msg_type: Optional[str] = None,
            flat_mode: bool = True,
            **kwargs
    ) -> Optional[str]:
        """
        Serializza dati con metadati opzionali in formato RTD-compatible.

        Regole:
        1. Se data è dict/Series con valori scalari → JSON piatto con metadati in __metadata__
        2. Se data è DataFrame → formato 'records' + metadati in __metadata__
        3. Se data è scalar/list → wrappa in struttura con __value__ e __metadata__

        Args:
            data: Dati da serializzare (già normalizzati)
            metadata: Metadati opzionali da includere
            msg_type: Tipo messaggio (es. 'DATA', 'COMMAND')
            flat_mode: Se True, usa formato piatto; se False, usa wrapping standard

        Returns:
            JSON string

        Examples:
            >>> # Dict con valori scalari → piatto
            >>> serialize_with_metadata(
            ...     {"AAPL": 150, "GOOGL": 2800},
            ...     metadata={"source": "bloomberg"},
            ...     msg_type="DATA"
            ... )
            {
                "AAPL": 150,
                "GOOGL": 2800,
                "__metadata__": {
                    "type": "DATA",
                    "source": "bloomberg"
                }
            }

            >>> # DataFrame → records + metadata
            >>> serialize_with_metadata(
            ...     [{"ticker": "AAPL", "price": 150}, {"ticker": "GOOGL", "price": 2800}],
            ...     metadata={"source": "bloomberg"},
            ...     msg_type="DATA"
            ... )
            {
                "records": [
                    {"ticker": "AAPL", "price": 150},
                    {"ticker": "GOOGL", "price": 2800}
                ],
                "__metadata__": {
                    "type": "DATA",
                    "source": "bloomberg"
                }
            }

            >>> # Scalar → wrapped
            >>> serialize_with_metadata(
            ...     42,
            ...     metadata={"unit": "seconds"},
            ...     msg_type="DATA"
            ... )
            {
                "__value__": 42,
                "__metadata__": {
                    "type": "DATA",
                    "unit": "seconds"
                }
            }
        """
        if data is None:
            return None

        if not flat_mode:
            # Modalità wrapped standard (backward compatibility)
            payload = {"data": data, "type": msg_type}
            if metadata:
                payload.update(metadata)
            return json.dumps(payload, default=str, date_format=kwargs.get('date_format', 'iso'))

        # Modalità flat RTD-compatible
        result = {}

        # Costruisci metadata se necessario
        meta = {}
        if msg_type:
            meta["type"] = msg_type
        if metadata:
            meta.update(metadata)

        # Determina struttura basata sul tipo di data
        if isinstance(data, dict):
            # Dict → merge diretto (valori scalari piatti)
            result.update(data)
            if meta:
                result["__metadata__"] = meta

        elif isinstance(data, list):
            # List (es. DataFrame records) → chiave 'records'
            result["records"] = data
            if meta:
                result["__metadata__"] = meta

        else:
            # Scalar o altri tipi → wrappa in __value__
            result["__value__"] = data
            if meta:
                result["__metadata__"] = meta

        try:
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Errore serializzazione con metadata: {e}")
            return None


# ============================================================================
# Main RedisPublisher gui Class
# ============================================================================

class RedisPublisher:
    """gui estesa con supporto RedisPublisher Pub/Sub per comunicazione in tempo reale."""

    def __init__(
            self,
            *args,
            redis_host: str = 'localhost',
            redis_port: int = 6379,
            redis_db: int = 0,
            **kwargs
    ):
        """
        Inizializza gui con funzionalità RedisPublisher.

        Args:
            redis_host: Host del server RedisPublisher
            redis_port: Porta del server RedisPublisher
            redis_db: Database RedisPublisher da utilizzare
        """
        super().__init__(*args, **kwargs)

        # RedisPublisher connection
        self.redis_client = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.redis_client.config_set('notify-keyspace-events', 'KEA')

        # Internal components
        self.normalizer = DataNormalizer()
        self.change_detector = ChangeDetector()
        self.serializer = DataSerializer()

        # Tracking
        self.available_channels: Set[str] = set()

    def export_data(self, skip_if_unchanged: bool = False, **data) -> None:
        """
        Pubblica dati su RedisPublisher usando Pub/Sub.

        Args:
            skip_if_unchanged: Se True, pubblica solo dati modificati (diff granulare per dict)
            **data: Coppie channel-value da pubblicare

        Examples:
            >>> # Pubblica sempre (comportamento default)
            >>> gui.export_data(prices={"AAPL": 150, "GOOGL": 2800})

            >>> # Pubblica solo se cambiato (per dict, solo chiavi modificate)
            >>> gui.export_data(skip_if_unchanged=True, prices={"AAPL": 150, "GOOGL": 2801})
            >>> # Pubblicherà solo {"GOOGL": 2801}
        """
        for channel, value in data.items():
            if value is None:
                continue

            try:
                # Step 1: Normalizza il dato
                normalized_value = self.normalizer.normalize(value)
                if normalized_value is None:
                    continue

                # Step 2: Rileva cambiamenti
                has_changed, value_to_publish = self.change_detector.has_changed(
                    channel,
                    normalized_value,
                    skip_if_unchanged
                )

                if not has_changed:
                    continue

                # Step 3: Serializza
                json_value = self.serializer.serialize(value_to_publish)
                if json_value is None:
                    continue

                # Step 4: Pubblica
                self.redis_client.publish(channel, json_value)
                self.available_channels.add(channel)
                logger.debug(f"Pubblicato su '{channel}' (skip_if_unchanged={skip_if_unchanged})")

            except redis.RedisError:
                logger.exception(f"Errore RedisPublisher su canale '{channel}'")
            except Exception:
                logger.exception(f"Errore pubblicazione '{channel}'")

    def export_static_data(self, **data) -> None:
        """
        Salva dati statici su RedisPublisher usando SET.

        Args:
            **data: Coppie key-value da salvare
        """
        for key, value in data.items():
            if value is None:
                continue

            try:
                # Normalizza e serializza
                normalized_value = self.normalizer.normalize(value)
                if normalized_value is None:
                    continue

                json_value = self.serializer.serialize(normalized_value)
                if json_value is None:
                    continue

                self.redis_client.set(key, json_value)
                logger.debug(f"Salvato '{key}'")

            except redis.RedisError:
                logger.exception(f"Errore RedisPublisher su chiave '{key}'")
            except Exception:
                logger.exception(f"Errore serializzazione '{key}'")

    def get_static_data(self, key: str) -> Optional[Any]:
        """
        Recupera dati statici da RedisPublisher.

        Args:
            key: Chiave da recuperare

        Returns:
            Valore deserializzato o None
        """
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except redis.RedisError:
            logger.exception(f"Errore RedisPublisher recupero '{key}'")
            return None
        except json.JSONDecodeError:
            logger.exception(f"Errore deserializzazione '{key}'")
            return None

    def clear_change_cache(self, channel: Optional[str] = None) -> None:
        """
        Pulisce la cache dei cambiamenti (utile per reset o debug).

        Args:
            channel: Canale specifico da pulire (None = pulisce tutto)
        """
        self.change_detector.clear_cache(channel)


# ============================================================================
# Extended RedisPublisher gui with Messaging Support
# ============================================================================

class RedisMessaging(RedisPublisher):
    """gui RedisPublisher con supporto per messaggi tipizzati (DATA/COMMAND/FLOW_DETECTED)."""

    def export_message(
            self,
            channel: str,
            value: Any,
            msg_type: MessageType = MessageType.DATA,
            skip_if_unchanged: bool = False,
            metadata: Optional[Dict[str, Any]] = None,
            flat_mode: bool = True,
            **kwargs
    ) -> None:
        """
        Pubblica un messaggio con metadati in formato RTD-compatible.

        NON USA PIÙ GUIMessage.to_json() - usa direttamente DataSerializer.

        Args:
            channel: Canale di pubblicazione
            value: Contenuto del messaggio
            msg_type: Tipo di messaggio (DATA, COMMAND, FLOW_DETECTED)
            skip_if_unchanged: Se True, usa change detection
            metadata: Metadati aggiuntivi da includere in __metadata__
            flat_mode: Se True (default), usa formato piatto RTD-compatible

        Examples:
            >>> # Prezzi (dict) → JSON piatto
            >>> gui.export_message("rtd:prices", {"AAPL": 150, "GOOGL": 2800})
            >>> # Pubblica: {"AAPL": 150, "GOOGL": 2800, "__metadata__": {"type": "DATA"}}

            >>> # DataFrame → records format
            >>> gui.export_message("rtd:positions", df)
            >>> # Pubblica: {"records": [...], "__metadata__": {"type": "DATA"}}

            >>> # Con metadata custom
            >>> gui.export_message(
            ...     "rtd:prices",
            ...     {"AAPL": 150},
            ...     metadata={"source": "bloomberg", "timestamp": "2025-12-16T10:30:00"}
            ... )
        """
        if value is None:
            return

        try:
            # Normalizza il valore
            normalized_value = self.normalizer.normalize(value, **kwargs)
            if normalized_value is None:
                return

            # ✅ USA DIRETTAMENTE DataSerializer invece di GUIMessage
            json_msg = self.serializer.serialize_with_metadata(data=normalized_value, metadata=metadata,
                                                               msg_type=msg_type.value, flat_mode=flat_mode,
                                                               **kwargs)

            if json_msg is None:
                return

            # Pubblica (bypass export_data per evitare doppia serializzazione)
            if skip_if_unchanged:
                # Usa change detection
                has_changed, _ = self.change_detector.has_changed(
                    channel,
                    json_msg,  # Confronta il JSON string direttamente
                    skip_if_unchanged
                )
                if not has_changed:
                    return

            self.redis_client.publish(channel, json_msg)
            self.available_channels.add(channel)
            logger.debug(f"Pubblicato messaggio su '{channel}' (flat_mode={flat_mode})")
        except Exception:
            logger.exception(f"Errore creazione messaggio per '{channel}'")

    def export_flow_detected(
            self,
            channel: str,
            flow: Flow,
            skip_if_unchanged: bool = False
    ) -> None:
        """
        Shortcut per pubblicare flow detection.

        Args:
            channel: Canale di pubblicazione
            flow: Oggetto Flow da pubblicare
            skip_if_unchanged: Se True, usa change detection
        """
        self.export_message(
            channel,
            flow.to_card_data(),
            MessageType.FLOW_DETECTED,
            skip_if_unchanged
        )

    def post_on_channel(self, skip_if_unchanged: bool = False, **kwargs) -> None:
        """
        Pubblica dati su canali multipli.

        Args:
            skip_if_unchanged: Se True, usa change detection
            **kwargs: Coppie channel=data da pubblicare
        """
        self.export_data(skip_if_unchanged=skip_if_unchanged, **kwargs)

    # ========================================================================
    # Store-aware publishing (con auto-timestamp per eventi)
    # ========================================================================

    def publish(
            self,
            channel: str,
            data: Any,
            store_type: str = "state",
            skip_if_unchanged: Optional[bool] = None,
            auto_timestamp: bool = False
    ) -> None:
        """
        Pubblica dati con semantica store-aware.

        Args:
            channel: RedisPublisher channel
            data: Dati da pubblicare
            store_type: Tipo di store ("state", "event", "blob")
                - "state": Default skip=True (dati che evolvono)
                - "event": Default skip=False (eventi unici)
                - "blob": Default skip=False (dati pesanti)
            skip_if_unchanged: Override del default basato su store_type
            auto_timestamp: Se True, aggiunge timestamp automaticamente (utile per eventi)

        Examples:
            >>> # State (default skip=True)
            >>> gui.publish("portfolio:positions", positions, store_type="state")

            >>> # Event (default skip=False, auto-timestamp)
            >>> gui.publish("events:trades", trade_data,
            ...            store_type="event", auto_timestamp=True)

            >>> # Override default
            >>> gui.publish("prices", prices, store_type="state", skip_if_unchanged=False)
        """
        if data is None:
            return

        try:
            # Default skip basati su store_type
            if skip_if_unchanged is None:
                skip_if_unchanged = (store_type == "state")

            # Auto-add timestamp se richiesto
            if auto_timestamp and isinstance(data, dict):
                if "timestamp" not in data:
                    from datetime import datetime
                    data["timestamp"] = datetime.now().isoformat()

            self.export_data(skip_if_unchanged=skip_if_unchanged, **{channel: data})

        except Exception as e:
            logger.error(f"Error publishing to '{channel}' (store_type={store_type}): {e}")
