import json
import logging
from datetime import datetime
from threading import Thread
from typing import Callable, Dict, Optional

import redis
from redis import StrictRedis

logger = logging.getLogger(__name__)


class CommandListener:
    """
    Ascolta comandi su un canale Redis pub/sub e invoca una callback.

    Usage:
        listener = CommandListener(redis_client, on_command=strategy._handle_command)
        listener.start()

    Trigger:
        redis-cli PUBLISH engine:commands '{"action": "reload_beta"}'

    Response (su engine:status):
        {"action": "reload_beta", "status": "success", "elapsed_seconds": 1.23, "timestamp": "..."}
    """

    def __init__(
        self,
        on_command: Callable[[str, dict], None],
        redis_client: redis.StrictRedis = None,
        channel: str = "engine:commands",
        status_channel: str = "engine:status",
    ):
        """
        Args:
            redis_client: Connessione Redis esistente
            on_command: Callback con signature (action: str, payload: dict) -> None
            channel: Canale Redis su cui ascoltare i comandi
            status_channel: Canale Redis su cui pubblicare lo status
        """
        self.redis_client = redis_client or StrictRedis()
        self.on_command = on_command
        self.channel = channel
        self.status_channel = status_channel
        self._pubsub: Optional[redis.client.PubSub] = None
        self._thread: Optional[Thread] = None
        self._running = False

    def start(self) -> None:
        """Avvia il listener in un daemon thread."""
        if self._running:
            logger.warning("CommandListener is already running")
            return

        self._pubsub = self.redis_client.pubsub()
        self._pubsub.subscribe(self.channel)
        self._running = True

        self._thread = Thread(target=self._listen_loop, daemon=True, name="CommandListener")
        self._thread.start()

        logger.info(f"CommandListener started on channel '{self.channel}'")

    def stop(self) -> None:
        """Ferma il listener."""
        self._running = False
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.debug(f"Error closing pubsub: {e}")
            self._pubsub = None
        self._thread = None
        logger.info("CommandListener stopped")

    def _listen_loop(self) -> None:
        """Loop principale del listener (gira nel daemon thread)."""
        try:
            for message in self._pubsub.listen():
                if not self._running:
                    break

                if message["type"] != "message":
                    continue

                try:
                    self._on_message(message["data"])
                except Exception as e:
                    logger.error(f"Error processing command: {e}", exc_info=True)

        except redis.ConnectionError as e:
            logger.error(f"CommandListener lost Redis connection: {e}")
        except Exception as e:
            logger.error(f"CommandListener unexpected error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("CommandListener loop exited")

    def _on_message(self, data: str) -> None:
        """Parsa il messaggio JSON, invoca on_command, pubblica status."""
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid command (not JSON): {data!r}")
            return

        action = payload.get("action")
        if not action:
            logger.warning(f"Command missing 'action' field: {payload}")
            return

        logger.info(f"Executing command: '{action}'")
        start = datetime.now()
        try:
            self.on_command(action, payload)
            elapsed = (datetime.now() - start).total_seconds()
            self._publish_status(action, "success", elapsed_seconds=elapsed)
            logger.info(f"Command '{action}' completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            self._publish_status(action, "error", error=str(e), elapsed_seconds=elapsed)
            logger.error(f"Command '{action}' failed after {elapsed:.2f}s: {e}", exc_info=True)

    def _publish_status(self, action: str, status: str,
                        error: Optional[str] = None,
                        elapsed_seconds: Optional[float] = None) -> None:
        """Pubblica il risultato del comando su engine:status."""
        response = {
            "action": action,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if error is not None:
            response["error"] = error
        if elapsed_seconds is not None:
            response["elapsed_seconds"] = round(elapsed_seconds, 3)

        try:
            self.redis_client.publish(self.status_channel, json.dumps(response))
        except Exception as e:
            logger.error(f"Failed to publish status: {e}")

    @property
    def is_running(self) -> bool:
        return self._running
