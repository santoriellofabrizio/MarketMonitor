"""
GUIQueue - Wrapper for exporting data to gui.
Enhanced with generic event export.
"""
from queue import Queue
import pandas as pd
from typing import Any, Dict, Optional

from market_monitor.publishers.base import MessageType


class GUIQueue:
    """
    Helper class to export data to gui via Queue.
    Wraps data in message format with type information.
    """

    def __init__(self, queue: Queue, auto_clean: bool = False):
        """
        Args:
            queue: Python Queue for communication
            auto_clean: If True, periodically clean old messages (not implemented)
        """
        self.queue = queue
        self.auto_clean = auto_clean

    def export_data(self, data: pd.DataFrame, **kwargs):
        """
        Export DataFrame as DATA message.

        Args:
            data: DataFrame to export
            **kwargs: Additional parameters to include in message
        """
        message = {
            'type': MessageType.DATA,
            'data': data,
            **kwargs
        }
        self.queue.put(message)

    def export_event(self, event_type: str, event_data: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Export a generic event message.

        This is the main method for sending events to the dashboard.

        Args:
            event_type: Type of event (e.g., 'flow_detected', 'alert', 'signal')
            event_data: Dictionary with event-specific data
            **kwargs: Additional fields to include in message

        Examples:
            # Flow detection
            gui.export_event('flow_detected', {
                'flow_id': 'flow_1',
                'side': 'buy',
                'ticker': 'AAPL',
                'total_quantity': 500.0,
                'consistency_score': 0.95
            })

            # Custom alert
            gui.export_event('alert', {
                'level': 'high',
                'message': 'Large trade detected',
                'ticker': 'MSFT'
            })

            # Trading signal
            gui.export_event('signal', {
                'signal_type': 'entry',
                'ticker': 'GOOGL',
                'price': 150.0
            })
        """
        message = {
            'type': MessageType.EVENT,
            **(event_data or {}),
            **kwargs
        }
        self.queue.put(message)

    def export_command(self, command: str, **kwargs):
        """
        Export a command message.

        Args:
            command: Command string
            **kwargs: Command parameters
        """
        message = {
            'type': MessageType.COMMAND,
            'command': command,
            **kwargs
        }
        self.queue.put(message)

    def export_status(self, status: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Export status information.

        Args:
            status: Status dictionary
            **kwargs: Additional status fields
        """
        message = {
            'type': MessageType.STATUS,
            'status': status or {},
            **kwargs
        }
        self.queue.put(message)

    def export_config(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Export configuration.

        Args:
            config: Configuration dictionary
            **kwargs: Additional config fields
        """
        message = {
            'type': MessageType.CONFIG,
            'config': config or {},
            **kwargs
        }
        self.queue.put(message)

    def export_error(self, error: str, **kwargs):
        """
        Export error message.

        Args:
            error: Error description
            **kwargs: Additional error information
        """
        message = {
            'type': MessageType.ERROR,
            'error': error,
            **kwargs
        }
        self.queue.put(message)

    # Convenience methods (optional - use export_event instead)

    def export_flow(self, flow_data: Dict[str, Any], **kwargs):
        """
        Convenience method for flow detection events.
        Equivalent to: export_event('flow_detected', flow_data, **kwargs)

        Args:
            flow_data: Flow information dictionary
            **kwargs: Additional fields
        """
        self.export_event('flow_detected', flow_data, **kwargs)

    def export_alert(self, alert_data: Dict[str, Any], **kwargs):
        """
        Convenience method for alert events.
        Equivalent to: export_event('alert', alert_data, **kwargs)

        Args:
            alert_data: Alert information dictionary
            **kwargs: Additional fields
        """
        self.export_event('alert', alert_data, **kwargs)

    def export_signal(self, signal_data: Dict[str, Any], **kwargs):
        """
        Convenience method for trading signal events.
        Equivalent to: export_event('signal', signal_data, **kwargs)

        Args:
            signal_data: Signal information dictionary
            **kwargs: Additional fields
        """
        self.export_event('signal', signal_data, **kwargs)