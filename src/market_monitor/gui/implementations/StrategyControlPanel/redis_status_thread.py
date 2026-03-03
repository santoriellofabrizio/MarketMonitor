"""
Background thread that subscribes to a Redis channel and emits Qt signals.
Used by StrategyControlPanel to receive command responses from engine:status.
"""
import json
import logging

import redis
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class RedisStatusThread(QThread):
    """
    Subscribes to a Redis pub/sub channel and emits a Qt signal for each message.

    Runs in a QThread so the main Qt event loop is never blocked.

    Signals:
        status_received (dict): Emitted with the parsed JSON payload for each
            message received on the subscribed channel.
        connection_error (str): Emitted when the Redis connection is lost.
    """

    status_received = pyqtSignal(dict)
    connection_error = pyqtSignal(str)

    def __init__(self, redis_client: redis.StrictRedis, channel: str, parent=None):
        super().__init__(parent)
        self._redis_client = redis_client
        self._channel = channel
        self._running = False
        self._pubsub = None

    def run(self) -> None:
        self._running = True
        try:
            self._pubsub = self._redis_client.pubsub()
            self._pubsub.subscribe(self._channel)
            logger.info(f"RedisStatusThread subscribed to '{self._channel}'")

            for message in self._pubsub.listen():
                if not self._running:
                    break
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                    self.status_received.emit(data)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"RedisStatusThread: invalid JSON: {e}")

        except redis.ConnectionError as e:
            logger.error(f"RedisStatusThread lost connection: {e}")
            self.connection_error.emit(str(e))
        except Exception as e:
            logger.error(f"RedisStatusThread unexpected error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("RedisStatusThread stopped")

    def stop(self) -> None:
        """Stop the listener loop and unsubscribe."""
        self._running = False
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.debug(f"RedisStatusThread stop error: {e}")
            self._pubsub = None
