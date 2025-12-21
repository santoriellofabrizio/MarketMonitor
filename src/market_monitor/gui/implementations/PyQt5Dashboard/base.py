"""
Base class for PyQt5Dashboard dashboards with data update loop.
"""
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QStatusBar
from PyQt5.QtCore import pyqtSignal, QObject
import pandas as pd
from typing import Optional
import logging

from market_monitor.publishers.base import MessageType


class DataSignals(QObject):
    """Qt Signals for data updates"""
    data_updated = pyqtSignal(pd.DataFrame)
    command_received = pyqtSignal(dict)
    config_updated = pyqtSignal(dict)
    status_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)


class BasePyQt5Dashboard(QMainWindow):
    """
    Base class for PyQt5Dashboard dashboards with automatic data update loop.
    """
    
    def __init__(self,
                 datasource,
                 update_interval_ms: int = 100,
                 title: str = "Dashboard",
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            datasource: DataStore instance
            update_interval_ms: Update interval in milliseconds
            title: Window title
            logger: Optional logger
        """
        super().__init__()

        self.datasource = datasource
        self.update_interval_ms = update_interval_ms
        self.paused = False
        self.logger = logger or logging.getLogger(__name__)
        
        # Signals
        self.signals = DataSignals()
        
        # Setup window
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # Register handlers
        self._register_handlers()
        
        # Setup UI (to be implemented by subclasses)
        self.setup_ui()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_count = 0
    
    def _register_handlers(self):
        """Register handlers for different message types"""
        if self.datasource and hasattr(self.datasource, "register_handler"):
            self.datasource.register_handler(MessageType.DATA, self._handle_data)
            self.datasource.register_handler(MessageType.COMMAND, self._handle_command)
            self.datasource.register_handler(MessageType.CONFIG, self._handle_config)
            self.datasource.register_handler(MessageType.STATUS, self._handle_status)
            self.datasource.register_handler(MessageType.ERROR, self._handle_error)
    
    def _handle_data(self, message: dict):
        """Handle DATA message"""
        df = message.get('data')
        if df is not None and not df.empty:
            self.signals.data_updated.emit(df)
    
    def _handle_command(self, message: dict):
        """Handle COMMAND message"""
        self.signals.command_received.emit(message)
    
    def _handle_config(self, message: dict):
        """Handle CONFIG message"""
        config = message.get('config', {})
        self.signals.config_updated.emit(config)
    
    def _handle_status(self, message: dict):
        """Handle STATUS message"""
        status = message.get('status', {})
        self.signals.status_updated.emit(status)
    
    def _handle_error(self, message: dict):
        """Handle ERROR message"""
        error = message.get('error', 'Unknown error')
        self.signals.error_occurred.emit(error)
    
    def _update_data(self):
        """Called by timer to check for new data"""
        if self.paused:
            return
        
        try:
            df = self.datasource.get_data()
            
            if not df.empty:
                self._update_count += len(df)
                self._update_status_bar()
        
        except Exception as e:
            self.logger.error(f"Error updating data: {e}")
    
    def _update_status_bar(self):
        """Update status bar with metrics"""
        status_text = f"Updates: {self._update_count} | "
        status_text += f"{'PAUSED' if self.paused else 'RUNNING'} | "
        status_text += f"{self.datasource.__class__.__name__}"
        self.status_bar.showMessage(status_text)
    
    def setup_ui(self):
        """Setup UI - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement setup_ui()")
    
    def pause(self):
        """Pause data updates"""
        self.paused = True
        self._update_status_bar()
        self.logger.info("Dashboard paused")
    
    def resume(self):
        """Resume data updates"""
        self.paused = False
        self._update_status_bar()
        self.logger.info("Dashboard resumed")
    
    def clear(self):
        """Clear all data - to be implemented by subclasses"""
        self._update_count = 0
        self._update_status_bar()
