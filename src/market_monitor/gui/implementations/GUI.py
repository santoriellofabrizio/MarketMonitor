from abc import ABC, abstractmethod
from typing import Any, Union, Dict


class GUI(ABC):

    """ Abstract class for gui's. Users gui should implement this interface, and its method export_data."""

    @abstractmethod
    def export_data(self, *args, **kwargs) -> None:
        pass

    def async_export_data(self, output):
        pass

    def close(self):
        pass


class GUIDummy(GUI):
    """
    This class represents a gui element, that can be used to print out output_NAV to the console
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output_lines = 0

    def export_data(self, *args, **kwargs):
        """ Exports the DataFrame to the terminal while keeping output_NAV on the same lines. """
        pass