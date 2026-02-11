import json
import logging
from typing import Any, Dict, Optional, Set

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from market_monitor.publishers.base import MessageType
from market_monitor.publishers.redis_publisher import (
    DataNormalizer,
    ChangeDetector,
    DataSerializer,
)

logger = logging.getLogger(__name__)

DEFAULT_RABBIT_HOST = 'rabbitmq.af.tst'
DEFAULT_RABBIT_PORT = 5672
DEFAULT_RABBIT_USER = 'mqclient'
DEFAULT_RABBIT_PASSWORD = 'Mqclient-00'
DEFAULT_RABBIT_VHOST = '/'


# ============================================================================
# Main RabbitPublisher Class
# ============================================================================

class RabbitPublisher:
    """
    Publisher per RabbitMQ con interfaccia analoga a RedisPublisher.

    Usa exchange di tipo 'fanout' per replicare il comportamento
    Pub/Sub di Redis: ogni consumer registrato sull'exchange riceve
    tutti i messaggi pubblicati.

    Differenze rispetto a Redis:
    - Non supporta export_static_data / get_static_data (RabbitMQ è un
      message broker, non un key-value store).
    - La connessione è persistente; se cade viene riconnessa automaticamente
      alla prima pubblicazione successiva.
    - Ogni "canale" Redis corrisponde a un exchange fanout RabbitMQ.
    """

    def __init__(
            self,
            *args,
            rabbit_host: str = DEFAULT_RABBIT_HOST,
            rabbit_port: int = DEFAULT_RABBIT_PORT,
            rabbit_user: str = DEFAULT_RABBIT_USER,
            rabbit_password: str = DEFAULT_RABBIT_PASSWORD,
            rabbit_vhost: str = DEFAULT_RABBIT_VHOST,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._rabbit_params = pika.ConnectionParameters(
            host=rabbit_host,
            port=rabbit_port,
            virtual_host=rabbit_vhost,
            credentials=pika.PlainCredentials(rabbit_user, rabbit_password),
            heartbeat=60,
            blocked_connection_timeout=30,
        )

        self._connection: Optional[pika.BlockingConnection] = None
        self._channel = None

        # Componenti interni (riuso da redis_publisher)
        self.normalizer = DataNormalizer()
        self.change_detector = ChangeDetector()
        self.serializer = DataSerializer()

        # Tracking
        self.available_channels: Set[str] = set()
        self._declared_exchanges: Set[str] = set()

        self._connect()

    # ------------------------------------------------------------------
    # Gestione connessione
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Stabilisce connessione e canale RabbitMQ."""
        try:
            self._connection = pika.BlockingConnection(self._rabbit_params)
            self._channel = self._connection.channel()
            self._declared_exchanges.clear()
            logger.info("Connessione RabbitMQ stabilita")
        except AMQPConnectionError as e:
            logger.error(f"Impossibile connettersi a RabbitMQ: {e}")
            self._connection = None
            self._channel = None

    def _ensure_connected(self) -> bool:
        """Verifica la connessione; tenta riconnessione se necessario."""
        if self._channel and self._channel.is_open:
            return True
        logger.warning("RabbitMQ non connesso, tentativo di riconnessione...")
        self._connect()
        return self._channel is not None and self._channel.is_open

    def _declare_exchange(self, exchange_name: str) -> None:
        """
        Dichiara un exchange fanout (operazione idempotente).
        Gli exchange sono durable per sopravvivere al restart del broker.
        """
        if exchange_name not in self._declared_exchanges:
            self._channel.exchange_declare(
                exchange=exchange_name,
                exchange_type='fanout',
                durable=True,
            )
            self._declared_exchanges.add(exchange_name)

    # ------------------------------------------------------------------
    # API pubblica (speculare a RedisPublisher)
    # ------------------------------------------------------------------

    def export_data(self, skip_if_unchanged: bool = False, **data) -> None:
        """
        Pubblica dati su RabbitMQ usando exchange fanout.

        Args:
            skip_if_unchanged: Se True, pubblica solo dati modificati
                               (diff granulare per dizionari).
            **data: Coppie exchange_name=value da pubblicare.

        Examples:
            >>> pub.export_data(trades_df=df)
            >>> pub.export_data(skip_if_unchanged=True, prices={"AAPL": 150})
        """
        for channel, value in data.items():
            if value is None:
                continue

            try:
                # Step 1: Normalizza
                normalized_value = self.normalizer.normalize(value)
                if normalized_value is None:
                    continue

                # Step 2: Change detection
                has_changed, value_to_publish = self.change_detector.has_changed(
                    channel, normalized_value, skip_if_unchanged
                )
                if not has_changed:
                    continue

                # Step 3: Serializza
                json_value = self.serializer.serialize(value_to_publish)
                if json_value is None:
                    continue

                # Step 4: Pubblica
                if not self._ensure_connected():
                    logger.error(
                        f"RabbitMQ non disponibile, messaggio perso su '{channel}'"
                    )
                    continue

                self._declare_exchange(channel)
                self._channel.basic_publish(
                    exchange=channel,
                    routing_key='',
                    body=json_value.encode('utf-8'),
                    properties=pika.BasicProperties(
                        delivery_mode=1,          # Non-persistent (performance)
                        content_type='application/json',
                    ),
                )
                self.available_channels.add(channel)
                logger.debug(f"Pubblicato su exchange '{channel}'")

            except (AMQPConnectionError, AMQPChannelError):
                logger.exception(
                    f"Errore RabbitMQ su '{channel}', tentativo riconnessione"
                )
                self._connect()
            except Exception:
                logger.exception(f"Errore pubblicazione '{channel}'")

    def export_static_data(self, **data) -> None:
        """
        Non supportato da RabbitMQ (broker, non key-value store).
        Logga un warning e non esegue operazioni.
        """
        logger.warning(
            "export_static_data non supportato da RabbitPublisher. "
            "Usare un datastore separato (Redis SET, database, ecc.)."
        )

    def get_static_data(self, key: str) -> Optional[Any]:
        """Non supportato da RabbitMQ. Ritorna None."""
        logger.warning(
            f"get_static_data('{key}') non supportato da RabbitPublisher."
        )
        return None

    def clear_change_cache(self, channel: Optional[str] = None) -> None:
        """
        Pulisce la cache dei cambiamenti.

        Args:
            channel: Canale specifico da pulire (None = pulisce tutto).
        """
        self.change_detector.clear_cache(channel)

    def close(self) -> None:
        """Chiude la connessione RabbitMQ in modo pulito."""
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
                logger.info("Connessione RabbitMQ chiusa")
        except Exception:
            logger.exception("Errore chiusura connessione RabbitMQ")


# ============================================================================
# Extended RabbitPublisher with Messaging Support
# ============================================================================

class RabbitMessaging(RabbitPublisher):
    """
    RabbitPublisher con supporto per messaggi tipizzati
    (DATA / COMMAND / FLOW_DETECTED).

    Interfaccia identica a RedisMessaging: drop-in replacement.
    """

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
        Pubblica un messaggio tipizzato su RabbitMQ in formato RTD-compatible.

        Args:
            channel: Exchange di pubblicazione (corrisponde al canale Redis).
            value: Contenuto del messaggio.
            msg_type: Tipo messaggio (DATA, COMMAND, FLOW_DETECTED, …).
            skip_if_unchanged: Se True, usa change detection.
            metadata: Metadati aggiuntivi in __metadata__.
            flat_mode: Se True (default), formato piatto RTD-compatible.

        Examples:
            >>> msg.export_message("trades_df", df, date_format='iso', orient="records")
        """
        if value is None:
            return

        try:
            normalized_value = self.normalizer.normalize(value, **kwargs)
            if normalized_value is None:
                return

            json_msg = self.serializer.serialize_with_metadata(
                data=normalized_value,
                metadata=metadata,
                msg_type=msg_type.value,
                flat_mode=flat_mode,
                **kwargs,
            )
            if json_msg is None:
                return

            if skip_if_unchanged:
                has_changed, _ = self.change_detector.has_changed(
                    channel, json_msg, skip_if_unchanged
                )
                if not has_changed:
                    return

            if not self._ensure_connected():
                logger.error(
                    f"RabbitMQ non disponibile, messaggio perso su '{channel}'"
                )
                return

            self._declare_exchange(channel)
            self._channel.basic_publish(
                exchange=channel,
                routing_key='',
                body=json_msg.encode('utf-8'),
                properties=pika.BasicProperties(
                    delivery_mode=1,
                    content_type='application/json',
                ),
            )
            self.available_channels.add(channel)
            logger.debug(
                f"Pubblicato messaggio tipizzato su '{channel}' "
                f"(type={msg_type.value}, flat_mode={flat_mode})"
            )

        except (AMQPConnectionError, AMQPChannelError):
            logger.exception(
                f"Errore RabbitMQ su '{channel}', tentativo riconnessione"
            )
            self._connect()
        except Exception:
            logger.exception(f"Errore creazione messaggio per '{channel}'")

    def export_flow_detected(
            self,
            channel: str,
            flow,
            skip_if_unchanged: bool = False
    ) -> None:
        """
        Shortcut per pubblicare flow detection.

        Args:
            channel: Exchange di pubblicazione.
            flow: Oggetto Flow con metodo to_card_data().
            skip_if_unchanged: Se True, usa change detection.
        """
        self.export_message(
            channel,
            flow.to_card_data(),
            MessageType.FLOW_DETECTED,
            skip_if_unchanged,
        )

    def post_on_channel(self, skip_if_unchanged: bool = False, **kwargs) -> None:
        """
        Pubblica dati su canali multipli.

        Args:
            skip_if_unchanged: Se True, usa change detection.
            **kwargs: Coppie channel=data.
        """
        self.export_data(skip_if_unchanged=skip_if_unchanged, **kwargs)

    def publish(
            self,
            channel: str,
            data: Any,
            store_type: str = "state",
            skip_if_unchanged: Optional[bool] = None,
            auto_timestamp: bool = False,
    ) -> None:
        """
        Pubblica dati con semantica store-aware (speculare a RedisMessaging.publish).

        Args:
            channel: Exchange di pubblicazione.
            data: Dati da pubblicare.
            store_type: "state" (default skip=True) | "event" | "blob".
            skip_if_unchanged: Override del default basato su store_type.
            auto_timestamp: Se True, aggiunge timestamp ai dict.
        """
        if data is None:
            return

        try:
            if skip_if_unchanged is None:
                skip_if_unchanged = (store_type == "state")

            if auto_timestamp and isinstance(data, dict):
                if "timestamp" not in data:
                    from datetime import datetime
                    data["timestamp"] = datetime.now().isoformat()

            self.export_data(
                skip_if_unchanged=skip_if_unchanged, **{channel: data}
            )

        except Exception as e:
            logger.error(
                f"Error publishing to '{channel}' (store_type={store_type}): {e}"
            )
