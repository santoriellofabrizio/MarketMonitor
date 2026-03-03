"""
StrategyControlPanel — PyQt5 control panel for any MarketMonitor strategy.

Provides four tabs:
  - Control  : strategy status + Stop button
  - Commands : form to send Redis commands + response log
  - Events   : real-time lifecycle event log
  - Logs     : Python logging stream

Integration (zero code required for existing strategies):
    Add to your YAML config:

        gui:
          control_panel:
            gui_type: "StrategyControlPanel"
            redis_config:
              host: localhost
              port: 6379
              db: 0
            commands_channel: "engine:commands"
            status_channel:   "engine:status"

Optional command registry (for richer Commands tab):
    Add to your strategy class:

        CONTROL_PANEL_COMMANDS = [
            {"action": "reload_beta", "description": "Reload beta matrices", "payload": {}},
            {"action": "update_forecaster", "description": "Update forecaster",
             "payload": {"model": "cluster", "type": "ewma_outlier", "params": {}}},
        ]
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import redis
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QSplitter, QListWidget, QListWidgetItem,
    QSizePolicy,
)

from market_monitor.gui.implementations.StrategyControlPanel.redis_status_thread import RedisStatusThread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qt signal bridge (must live in a QObject)
# ---------------------------------------------------------------------------

class _ControlPanelSignals(QObject):
    lifecycle_event = pyqtSignal(str, object)   # (event_name, data)
    status_update   = pyqtSignal(dict)           # response from engine:status
    log_message     = pyqtSignal(str, str)       # (level_name, formatted_line)


# ---------------------------------------------------------------------------
# Custom logging handler that routes records to the Qt log panel
# ---------------------------------------------------------------------------

class QTextEditLogger(logging.Handler):
    """
    Logging handler that emits log records through a Qt signal so they can
    be displayed safely in the main thread.
    """

    def __init__(self, signals: _ControlPanelSignals):
        super().__init__()
        self._signals = signals

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            self._signals.log_message.emit(record.levelname, line)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class StrategyControlPanel(QMainWindow):
    """
    PyQt5 control panel that works with *any* strategy.

    Implements the GUI interface via duck typing (export_data / close / start)
    to avoid the metaclass conflict between PyQt5's sip.wrappertype and ABCMeta.

    Register it via builder YAML or programmatically:

        strategy.set_gui("control", panel)
        panel.load_commands_from_strategy(strategy)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        redis_config: Optional[dict] = None,
        commands_channel: str = "engine:commands",
        status_channel: str = "engine:status",
        strategy_ref=None,
        **kwargs,
    ):
        QMainWindow.__init__(self)

        self._commands_channel = commands_channel
        self._status_channel = status_channel
        self._strategy = strategy_ref
        self._redis_client: Optional[redis.StrictRedis] = None
        self._status_thread: Optional[RedisStatusThread] = None
        self._log_handler: Optional[QTextEditLogger] = None
        self._log_level_filter: int = logging.DEBUG

        self._signals = _ControlPanelSignals()

        self._build_redis_client(redis_config or {})
        self._setup_ui()
        self._connect_signals()
        self._install_log_handler()

        if self._status_thread:
            self._status_thread.start()

        logger.info("StrategyControlPanel initialized")

    # ------------------------------------------------------------------
    # GUI interface
    # ------------------------------------------------------------------

    def export_data(self, **kwargs) -> None:
        """
        Called by the strategy for lifecycle event notifications.

        Expected kwargs:
            event_name (str): e.g. 'on_trade', 'on_book_initialized'
            data (Any): optional payload (e.g. trade count)
        """
        event_name = kwargs.get("event_name", "unknown")
        data = kwargs.get("data", None)
        self._signals.lifecycle_event.emit(event_name, data)

    def close(self) -> None:
        """Stop background threads and close the window."""
        self._uninstall_log_handler()
        if self._status_thread and self._status_thread.isRunning():
            self._status_thread.stop()
            self._status_thread.wait(2000)
        super().close()

    def start(self) -> None:
        """Show the window (called by builder after strategy threads are started)."""
        self.show()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_commands_from_strategy(self, strategy) -> None:
        """
        Read CONTROL_PANEL_COMMANDS from *strategy* (if defined) and populate
        the command dropdown with pre-filled entries.

        Safe to call even if the attribute is absent.
        """
        commands = getattr(strategy, "CONTROL_PANEL_COMMANDS", None)
        if not commands:
            return
        for cmd in commands:
            action = cmd.get("action", "")
            description = cmd.get("description", "")
            payload = cmd.get("payload", {})
            label = f"{action}" + (f"  —  {description}" if description else "")
            self._cmd_action_combo.addItem(label, userData=(action, payload))
        logger.debug(f"Loaded {len(commands)} commands from strategy")

    # ------------------------------------------------------------------
    # Redis client
    # ------------------------------------------------------------------

    def _build_redis_client(self, redis_config: dict) -> None:
        host = redis_config.get("host", "localhost")
        port = int(redis_config.get("port", 6379))
        db = int(redis_config.get("db", 0))
        try:
            self._redis_client = redis.StrictRedis(
                host=host, port=port, db=db,
                decode_responses=True, socket_connect_timeout=3,
            )
            self._redis_client.ping()
            logger.info(f"Redis connected: {host}:{port}/{db}")

            self._status_thread = RedisStatusThread(
                redis_client=self._redis_client,
                channel=self._status_channel,
            )
        except redis.RedisError as e:
            logger.warning(f"Redis not available: {e}. Commands tab will be disabled.")
            self._redis_client = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Strategy Control Panel")
        self.setGeometry(200, 200, 900, 650)
        self.setMinimumSize(700, 500)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # ---- header ----
        header = QHBoxLayout()
        title = QLabel("Strategy Control Panel")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        self._status_badge = QLabel("STOPPED")
        self._status_badge.setFixedWidth(140)
        self._status_badge.setAlignment(Qt.AlignCenter)
        self._set_status("STOPPED")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("Status:"))
        header.addWidget(self._status_badge)
        root_layout.addLayout(header)

        # ---- tabs ----
        self._tabs = QTabWidget()
        root_layout.addWidget(self._tabs)

        self._tabs.addTab(self._build_control_tab(),  "Control")
        self._tabs.addTab(self._build_commands_tab(), "Commands")
        self._tabs.addTab(self._build_events_tab(),   "Events")
        self._tabs.addTab(self._build_logs_tab(),     "Logs")

    # --- Control tab ---

    def _build_control_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)

        grp = QGroupBox("Strategy Control")
        grp_layout = QVBoxLayout(grp)

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("Current status:"))
        self._control_status_label = QLabel("STOPPED")
        self._control_status_label.setFont(QFont("Arial", 11, QFont.Bold))
        info_row.addWidget(self._control_status_label)
        info_row.addStretch()
        grp_layout.addLayout(info_row)

        self._last_event_label = QLabel("Last event: —")
        grp_layout.addWidget(self._last_event_label)

        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton("Stop Strategy")
        self._stop_btn.setFixedWidth(150)
        self._stop_btn.setStyleSheet("QPushButton { background-color: #c0392b; color: white; font-weight: bold; }")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)

        layout.addWidget(grp)
        layout.addStretch()
        return w

    # --- Commands tab ---

    def _build_commands_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        send_grp = QGroupBox("Send Command")
        send_layout = QVBoxLayout(send_grp)

        # Action row: combo (for CONTROL_PANEL_COMMANDS) + free text
        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Action:"))
        self._cmd_action_combo = QComboBox()
        self._cmd_action_combo.setEditable(True)
        self._cmd_action_combo.setMinimumWidth(260)
        self._cmd_action_combo.addItem("(free text)", userData=None)
        self._cmd_action_combo.currentIndexChanged.connect(self._on_command_selected)
        action_row.addWidget(self._cmd_action_combo)
        action_row.addStretch()
        send_layout.addLayout(action_row)

        # Payload editor
        send_layout.addWidget(QLabel("JSON Payload:"))
        self._cmd_payload_edit = QTextEdit()
        self._cmd_payload_edit.setPlaceholderText("{}")
        self._cmd_payload_edit.setMaximumHeight(100)
        self._cmd_payload_edit.setFont(QFont("Courier New", 9))
        send_layout.addWidget(self._cmd_payload_edit)

        # Send button
        btn_row = QHBoxLayout()
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(100)
        self._send_btn.setEnabled(self._redis_client is not None)
        self._send_btn.clicked.connect(self._on_send_command)
        self._cmd_error_label = QLabel("")
        self._cmd_error_label.setStyleSheet("color: red;")
        btn_row.addWidget(self._send_btn)
        btn_row.addWidget(self._cmd_error_label)
        btn_row.addStretch()
        send_layout.addLayout(btn_row)

        layout.addWidget(send_grp)

        # Response table
        layout.addWidget(QLabel("Command Responses:"))
        self._response_table = QTableWidget(0, 5)
        self._response_table.setHorizontalHeaderLabels(
            ["Timestamp", "Action", "Status", "Elapsed (s)", "Error"]
        )
        self._response_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._response_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._response_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self._response_table)

        return w

    # --- Events tab ---

    def _build_events_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("Lifecycle Events:"))
        btn_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(80)
        clear_btn.clicked.connect(lambda: self._event_list.clear())
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        self._event_list = QListWidget()
        self._event_list.setFont(QFont("Courier New", 9))
        layout.addWidget(self._event_list)

        return w

    # --- Logs tab ---

    def _build_logs_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Min level:"))
        self._log_level_combo = QComboBox()
        self._log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._log_level_combo.setCurrentText("DEBUG")
        self._log_level_combo.currentTextChanged.connect(self._on_log_level_changed)
        ctrl_row.addWidget(self._log_level_combo)
        ctrl_row.addStretch()
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setFixedWidth(80)
        clear_log_btn.clicked.connect(lambda: self._log_edit.clear())
        ctrl_row.addWidget(clear_log_btn)
        layout.addLayout(ctrl_row)

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFont(QFont("Courier New", 9))
        layout.addWidget(self._log_edit)

        return w

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._signals.lifecycle_event.connect(self._on_lifecycle_event)
        self._signals.log_message.connect(self._on_log_message)
        if self._status_thread:
            self._status_thread.status_received.connect(self._on_status_received)
            self._status_thread.connection_error.connect(self._on_redis_error)

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------

    def _on_lifecycle_event(self, event_name: str, data: Any) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Compose display text
        extra = ""
        if data is not None:
            if isinstance(data, int):
                extra = f" — {data} trade(s)"
            elif data:
                extra = f" — {data}"
        item_text = f"[{ts}]  {event_name}{extra}"
        item = QListWidgetItem(item_text)

        # Colour-code by event
        colour_map = {
            "on_start_strategy":    "#27ae60",
            "on_book_initialized":  "#2980b9",
            "on_stop":              "#c0392b",
            "on_trade":             "#8e44ad",
            "on_my_trade":          "#d35400",
        }
        colour = colour_map.get(event_name)
        if colour:
            item.setForeground(QColor(colour))

        self._event_list.addItem(item)
        self._event_list.scrollToBottom()
        self._last_event_label.setText(f"Last event: {event_name}  [{ts}]")

        # Update status badge based on event
        status_transitions = {
            "on_book_initialized": "BOOK READY",
            "on_start_strategy":   "RUNNING",
            "on_stop":             "STOPPED",
        }
        if event_name in status_transitions:
            self._set_status(status_transitions[event_name])

    def _on_status_received(self, data: dict) -> None:
        """Adds a row to the response table for each engine:status message."""
        row = self._response_table.rowCount()
        self._response_table.insertRow(row)

        ts = data.get("timestamp", "")
        if ts and "T" in ts:
            ts = ts.split("T")[1][:12]   # HH:MM:SS.mmm

        cols = [
            ts,
            data.get("action", ""),
            data.get("status", ""),
            str(round(data.get("elapsed_seconds", 0), 3)),
            data.get("error", ""),
        ]
        for col, val in enumerate(cols):
            cell = QTableWidgetItem(val)
            if data.get("status") == "error":
                cell.setForeground(QColor("#c0392b"))
            self._response_table.setItem(row, col, cell)

        self._response_table.scrollToBottom()

    def _on_log_message(self, level_name: str, line: str) -> None:
        level = getattr(logging, level_name, logging.DEBUG)
        if level < self._log_level_filter:
            return

        colour_map = {
            "DEBUG":    "#7f8c8d",
            "INFO":     "#2c3e50",
            "WARNING":  "#e67e22",
            "ERROR":    "#c0392b",
            "CRITICAL": "#922b21",
        }
        colour = colour_map.get(level_name, "#2c3e50")

        cursor = self._log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        cursor.mergeCharFormat(fmt)
        cursor.insertText(line + "\n")
        self._log_edit.setTextCursor(cursor)
        self._log_edit.ensureCursorVisible()

    def _on_redis_error(self, error: str) -> None:
        self._cmd_error_label.setText(f"Redis error: {error}")
        self._send_btn.setEnabled(False)

    def _on_stop_clicked(self) -> None:
        if self._strategy is not None:
            try:
                self._strategy.stop()
            except Exception as e:
                logger.error(f"Error stopping strategy: {e}")
        elif self._redis_client:
            try:
                self._redis_client.publish(
                    self._commands_channel,
                    json.dumps({"action": "stop"}),
                )
            except redis.RedisError as e:
                logger.error(f"Redis publish error on stop: {e}")

    def _on_send_command(self) -> None:
        self._cmd_error_label.setText("")
        action_text = self._cmd_action_combo.currentText().strip()
        payload_text = self._cmd_payload_edit.toPlainText().strip() or "{}"

        # If a predefined command is selected, use its stored action
        idx = self._cmd_action_combo.currentIndex()
        user_data = self._cmd_action_combo.itemData(idx)
        if user_data and isinstance(user_data, tuple):
            action, _default_payload = user_data
        else:
            action = action_text.split("  —  ")[0].strip()

        if not action:
            self._cmd_error_label.setText("Action is required")
            return

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as e:
            self._cmd_error_label.setText(f"Invalid JSON: {e}")
            return

        payload["action"] = action

        if not self._redis_client:
            self._cmd_error_label.setText("Redis not connected")
            return

        try:
            self._redis_client.publish(self._commands_channel, json.dumps(payload))
            logger.info(f"Command sent: {payload}")
        except redis.RedisError as e:
            self._cmd_error_label.setText(f"Send error: {e}")
            logger.error(f"Redis publish error: {e}")

    def _on_command_selected(self, index: int) -> None:
        """Pre-fill the payload editor when a predefined command is selected."""
        user_data = self._cmd_action_combo.itemData(index)
        if user_data and isinstance(user_data, tuple):
            _action, payload = user_data
            self._cmd_payload_edit.setText(
                json.dumps(payload, indent=2) if payload else "{}"
            )

    def _on_log_level_changed(self, level_name: str) -> None:
        self._log_level_filter = getattr(logging, level_name, logging.DEBUG)
        if self._log_handler:
            self._log_handler.setLevel(self._log_level_filter)

    # ------------------------------------------------------------------
    # Status badge
    # ------------------------------------------------------------------

    _STATUS_STYLES = {
        "STOPPED":    "background:#c0392b; color:white; border-radius:4px; padding:2px 8px; font-weight:bold;",
        "BOOK READY": "background:#e67e22; color:white; border-radius:4px; padding:2px 8px; font-weight:bold;",
        "RUNNING":    "background:#27ae60; color:white; border-radius:4px; padding:2px 8px; font-weight:bold;",
    }

    def _set_status(self, status: str) -> None:
        style = self._STATUS_STYLES.get(status, self._STATUS_STYLES["STOPPED"])
        self._status_badge.setText(status)
        self._status_badge.setStyleSheet(style)
        if hasattr(self, "_control_status_label"):
            self._control_status_label.setText(status)
            self._control_status_label.setStyleSheet(f"color: {self._badge_text_colour(status)};")

    @staticmethod
    def _badge_text_colour(status: str) -> str:
        return {"RUNNING": "#27ae60", "BOOK READY": "#e67e22"}.get(status, "#c0392b")

    # ------------------------------------------------------------------
    # Log handler management
    # ------------------------------------------------------------------

    def _install_log_handler(self) -> None:
        self._log_handler = QTextEditLogger(self._signals)
        self._log_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
                                datefmt="%H:%M:%S")
        self._log_handler.setFormatter(fmt)
        logging.getLogger().addHandler(self._log_handler)

    def _uninstall_log_handler(self) -> None:
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
