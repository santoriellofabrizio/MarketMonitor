import logging
from typing import Callable, Any, Dict
from pathlib import Path

from ruamel.yaml import YAML
from watchdog.events import FileSystemEventHandler

reader = YAML(typ='safe')
logger = logging.getLogger()


class ConfigChangeHandler(FileSystemEventHandler):
    """
    Handler per modifiche ai file di configurazione dinamica.
    Invoca callback quando vengono rilevati cambiamenti.
    """

    def __init__(self,
                 dynamic_config_path: str | Path,
                 on_config_change: Callable[[str, Any, Any], None]):
        """
        Args:
            dynamic_config_path: Path al file di configurazione da monitorare
            on_config_change: Callback con signature (key: str, old_value: Any, new_value: Any) -> None
        """
        super().__init__()
        self.dynamic_config_path = Path(dynamic_config_path)
        self.on_config_change = on_config_change
        self._last_config: Dict[str, Any] = {}

        self._load_initial_config()

    def _load_initial_config(self):
        """Carica la configurazione iniziale."""
        try:
            if self.dynamic_config_path.exists():
                with open(self.dynamic_config_path, 'r') as stream:
                    self._last_config = reader.load(stream) or {}
                logger.info(f"Loaded initial config: {list(self._last_config.keys())}")
        except Exception as e:
            logger.error(f"Failed to load initial config: {e}")
            self._last_config = {}

    def on_modified(self, event):
        """Chiamato quando il file viene modificato."""
        if event.src_path != str(self.dynamic_config_path):
            return

        try:
            with open(self.dynamic_config_path, 'r') as stream:
                new_config = reader.load(stream) or {}

            self._process_changes(new_config)
            self._last_config = new_config

        except Exception as e:
            logger.error(f"Failed to process config change: {e}", exc_info=True)

    def _process_changes(self, new_config: Dict[str, Any]):
        """Identifica e processa i cambiamenti."""
        all_keys = set(self._last_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_value = self._last_config.get(key)
            new_value = new_config.get(key)

            if old_value != new_value:
                logger.info(f"Config change detected: {key}: {old_value} -> {new_value}")

                try:
                    self.on_config_change(key, old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in callback for {key}: {e}", exc_info=True)