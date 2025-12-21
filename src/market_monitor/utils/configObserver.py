import logging
from typing import Optional, Any

from ruamel.yaml import YAML
from watchdog.events import FileSystemEventHandler


reader = YAML(typ='safe')

logger = logging.getLogger()
class ConfigChangeHandler(FileSystemEventHandler):

    """
    This class is used to handle modifications in dynamic config files. Modifies running strategy accordingly.
    """
    def __init__(self, dynamic_config_path, params=None):
        super().__init__()
        self.dynamic_config_path = dynamic_config_path
        self.params = params


    def on_modified(self, event):
        """
        This method is called when dynamic config file is modified. New content is read and validated.
        Args:
            event:

        Returns:

        """
        try:
            with open(self.dynamic_config_path, 'r') as stream:
                dynamic_params = reader.load(stream)

            for attr, new_value in dynamic_params.items():
                if hasattr(self.params, attr):
                    old_value = getattr(self.params, attr)
                    if old_value != new_value:
                        logger.info(f"{attr}: {old_value} -> {new_value}")
                        if isinstance(new_value, type(old_value)):
                            setattr(self.params, attr, new_value)
                        else:
                            logger.error(f"ERROR IN CONFIG CHANGE: {attr}: {type(new_value)} -> {type(old_value)}")
                else:
                    logger.warning(f"Attribute {attr} not found in market_monitor_fi.")
        except Exception as e:
            logger.error(f"Failed to process modification: {e}")


