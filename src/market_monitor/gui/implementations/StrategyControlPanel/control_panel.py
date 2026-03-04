"""
StrategyControlPanel — standalone PyQt5 control panel for MarketMonitor strategies.

Fully decoupled from strategy code. Communicates exclusively via Redis.
Can optionally launch a strategy as a subprocess.

Usage:
    run-control-panel [--host HOST] [--port PORT] [--config CONFIG_NAME]
                      [--lifecycle-channel engine:lifecycle]

Or connect to an already-running strategy:
    run-control-panel   (panel shows up; strategy running elsewhere)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import redis
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QProcess, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem,
)

from market_monitor.gui.implementations.StrategyControlPanel.redis_status_thread import RedisStatusThread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qt signal bridge (must live in a QObject to support pyqtSignal)
# ---------------------------------------------------------------------------

class _Signals(QObject):
    lifecycle_event = pyqtSignal(str, object)   # (event_name, data)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class StrategyControlPanel(QMainWindow):
    """
    Standalone PyQt5 control panel.

    Does NOT depend on StrategyUIAsync, Builder, or any strategy class.
    All communication with the strategy happens via Redis pub/sub:

        engine:commands   → sends user commands
        engine:status     → receives command responses
        engine:lifecycle  → receives lifecycle events (opt-in, see lifecycle_publisher task)

    The panel can also launch and kill a strategy subprocess via QProcess.
    """

    def __init__(
        self,
        redis_config: Optional[dict] = None,
        commands_channel: str = "engine:commands",
        status_channel: str = "engine:status",
        lifecycle_channel: Optional[str] = None,
        initial_config: Optional[str] = None,
        **kwargs,
    ):
        QMainWindow.__init__(self)

        self._commands_channel = commands_channel
        self._status_channel = status_channel
        self._lifecycle_channel = lifecycle_channel
        self._redis_client: Optional[redis.StrictRedis] = None
        self._status_thread: Optional[RedisStatusThread] = None
        self._lifecycle_thread: Optional[RedisStatusThread] = None
        self._process: Optional[QProcess] = None
        self._signals = _Signals()

        self._build_redis_client(redis_config or {})
        self._setup_ui(initial_config)
        self._connect_signals()

        if self._status_thread:
            self._status_thread.start()
        if self._lifecycle_thread:
            self._lifecycle_thread.start()

        logger.info("StrategyControlPanel initialized")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._stop_redis_threads()
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
        super().closeEvent(event)

    def _stop_redis_threads(self) -> None:
        for thread in (self._status_thread, self._lifecycle_thread):
            if thread and thread.isRunning():
                thread.stop()
                thread.wait(2000)

    # ------------------------------------------------------------------
    # Redis setup
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
            self._status_thread = RedisStatusThread(self._redis_client, self._status_channel)
            if self._lifecycle_channel:
                self._lifecycle_thread = RedisStatusThread(self._redis_client, self._lifecycle_channel)
        except redis.RedisError as e:
            logger.warning(f"Redis not available: {e}. Commands will be disabled.")
            self._redis_client = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self, initial_config: Optional[str] = None) -> None:
        self.setWindowTitle("Strategy Control Panel")
        self.setGeometry(200, 200, 950, 700)
        self.setMinimumSize(750, 520)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # Header
        header = QHBoxLayout()
        title_lbl = QLabel("Strategy Control Panel")
        title_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        self._status_badge = QLabel("STOPPED")
        self._status_badge.setFixedWidth(140)
        self._status_badge.setAlignment(Qt.AlignCenter)
        self._set_status("STOPPED")
        header.addWidget(title_lbl)
        header.addStretch()
        header.addWidget(QLabel("Status:"))
        header.addWidget(self._status_badge)
        root.addLayout(header)

        # Tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)
        self._tabs.addTab(self._build_control_tab(initial_config), "Control")
        self._tabs.addTab(self._build_commands_tab(),              "Commands")
        self._tabs.addTab(self._build_events_tab(),                "Events")
        self._tabs.addTab(self._build_logs_tab(),                  "Logs")

    # --- Control tab ---

    def _build_control_tab(self, initial_config: Optional[str] = None) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)

        # Launch group
        launch_grp = QGroupBox("Launch Strategy")
        launch_layout = QVBoxLayout(launch_grp)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Config:"))
        self._config_combo = QComboBox()
        self._config_combo.setEditable(True)
        self._config_combo.setMinimumWidth(320)
        self._populate_config_list(initial_config)
        config_row.addWidget(self._config_combo)
        config_row.addStretch()
        launch_layout.addLayout(config_row)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Strategy")
        self._start_btn.setFixedWidth(155)
        self._start_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#2ecc71;}"
        )
        self._start_btn.clicked.connect(self._on_start_clicked)

        self._stop_btn = QPushButton("Stop Strategy")
        self._stop_btn.setFixedWidth(155)
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;font-weight:bold;border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#e74c3c;}"
        )
        self._stop_btn.clicked.connect(self._on_stop_clicked)

        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        launch_layout.addLayout(btn_row)

        self._pid_label = QLabel("Process: not running")
        self._pid_label.setStyleSheet("color: #7f8c8d;")
        launch_layout.addWidget(self._pid_label)

        layout.addWidget(launch_grp)

        # Status group
        status_grp = QGroupBox("Strategy Status")
        status_layout = QVBoxLayout(status_grp)
        self._control_status_label = QLabel("STOPPED")
        self._control_status_label.setFont(QFont("Arial", 11, QFont.Bold))
        self._control_status_label.setStyleSheet("color: #c0392b;")
        self._last_event_label = QLabel("Last event: —")
        self._last_event_label.setStyleSheet("color: #7f8c8d;")
        status_layout.addWidget(self._control_status_label)
        status_layout.addWidget(self._last_event_label)
        layout.addWidget(status_grp)

        layout.addStretch()
        return w

    def _populate_config_list(self, initial_config: Optional[str] = None) -> None:
        try:
            from market_monitor.utils.config_helpers import find_all_configs
            for name in find_all_configs():
                self._config_combo.addItem(name)
        except Exception as e:
            logger.debug(f"Could not load config list: {e}")

        if initial_config:
            idx = self._config_combo.findText(initial_config)
            if idx >= 0:
                self._config_combo.setCurrentIndex(idx)
            else:
                self._config_combo.setCurrentText(initial_config)

    # --- Commands tab ---

    def _build_commands_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        send_grp = QGroupBox("Send Command")
        send_layout = QVBoxLayout(send_grp)

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

        send_layout.addWidget(QLabel("JSON Payload:"))
        self._cmd_payload_edit = QTextEdit()
        self._cmd_payload_edit.setPlaceholderText("{}")
        self._cmd_payload_edit.setMaximumHeight(100)
        self._cmd_payload_edit.setFont(QFont("Courier New", 9))
        send_layout.addWidget(self._cmd_payload_edit)

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
        ctrl_row.addWidget(QLabel("Strategy Process Output:"))
        ctrl_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(80)
        clear_btn.clicked.connect(lambda: self._log_edit.clear())
        ctrl_row.addWidget(clear_btn)
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
        if self._status_thread:
            self._status_thread.status_received.connect(self._on_status_received)
            self._status_thread.connection_error.connect(self._on_redis_connection_error)
        if self._lifecycle_thread:
            self._lifecycle_thread.status_received.connect(self._on_lifecycle_redis_message)
            self._lifecycle_thread.connection_error.connect(self._on_redis_connection_error)

    # ------------------------------------------------------------------
    # Slots — process management
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        config_name = self._config_combo.currentText().strip()
        if not config_name:
            self._log_edit.append("[ERROR] Select a config first.")
            return

        if self._process and self._process.state() != QProcess.NotRunning:
            self._log_edit.append("[WARN] Strategy already running — stop it first.")
            return

        self._log_edit.clear()
        self._log_edit.append(f"[INFO] Starting: run-strategy {config_name}\n")

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_process_output)
        self._process.finished.connect(self._on_process_finished)
        self._process.start("run-strategy", [config_name])

        if self._process.waitForStarted(3000):
            pid = self._process.processId()
            self._pid_label.setText(f"Process PID: {pid}")
            self._pid_label.setStyleSheet("color: #27ae60;")
            self._set_status("RUNNING")
        else:
            self._log_edit.append("[ERROR] Failed to start. Is run-strategy in PATH?")
            self._pid_label.setText("Process: failed to start")
            self._pid_label.setStyleSheet("color: #c0392b;")

    def _on_stop_clicked(self) -> None:
        if self._redis_client:
            try:
                self._redis_client.publish(
                    self._commands_channel, json.dumps({"action": "stop"})
                )
                self._log_edit.append("[INFO] Stop command sent via Redis.")
            except redis.RedisError as e:
                self._log_edit.append(f"[WARN] Redis error sending stop: {e}")

        if self._process and self._process.state() != QProcess.NotRunning:
            QTimer.singleShot(3000, self._kill_process_if_running)

    def _kill_process_if_running(self) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._log_edit.append("[INFO] Process killed (timeout).")
            self._pid_label.setText("Process: killed")
            self._pid_label.setStyleSheet("color: #c0392b;")

    def _on_process_output(self) -> None:
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        cursor = self._log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_edit.setTextCursor(cursor)
        self._log_edit.insertPlainText(data)
        self._log_edit.ensureCursorVisible()

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        self._log_edit.append(f"\n[INFO] Process finished (exit code: {exit_code})")
        self._pid_label.setText("Process: not running")
        self._pid_label.setStyleSheet("color: #7f8c8d;")
        self._set_status("STOPPED")

    # ------------------------------------------------------------------
    # Slots — Redis / lifecycle
    # ------------------------------------------------------------------

    def _on_lifecycle_event(self, event_name: str, data: Any) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        extra = ""
        if data is not None:
            extra = f" — {data} trade(s)" if isinstance(data, int) else f" — {data}"
        item = QListWidgetItem(f"[{ts}]  {event_name}{extra}")
        colour_map = {
            "on_start_strategy":   "#27ae60",
            "on_book_initialized": "#2980b9",
            "on_stop":             "#c0392b",
            "on_trade":            "#8e44ad",
            "on_my_trade":         "#d35400",
        }
        if colour := colour_map.get(event_name):
            item.setForeground(QColor(colour))
        self._event_list.addItem(item)
        self._event_list.scrollToBottom()
        self._last_event_label.setText(f"Last event: {event_name}  [{ts}]")
        status_map = {
            "on_book_initialized": "BOOK READY",
            "on_start_strategy":   "RUNNING",
            "on_stop":             "STOPPED",
        }
        if status := status_map.get(event_name):
            self._set_status(status)

    def _on_lifecycle_redis_message(self, data: dict) -> None:
        self._signals.lifecycle_event.emit(data.get("event", "unknown"), data.get("data"))

    def _on_status_received(self, data: dict) -> None:
        row = self._response_table.rowCount()
        self._response_table.insertRow(row)
        ts = data.get("timestamp", "")
        if ts and "T" in ts:
            ts = ts.split("T")[1][:12]
        cols = [
            ts, data.get("action", ""), data.get("status", ""),
            str(round(data.get("elapsed_seconds", 0), 3)), data.get("error", ""),
        ]
        for col, val in enumerate(cols):
            cell = QTableWidgetItem(val)
            if data.get("status") == "error":
                cell.setForeground(QColor("#c0392b"))
            self._response_table.setItem(row, col, cell)
        self._response_table.scrollToBottom()

    def _on_redis_connection_error(self, error: str) -> None:
        self._cmd_error_label.setText(f"Redis: {error}")
        self._send_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Slots — Commands tab
    # ------------------------------------------------------------------

    def _on_send_command(self) -> None:
        self._cmd_error_label.setText("")
        idx = self._cmd_action_combo.currentIndex()
        user_data = self._cmd_action_combo.itemData(idx)
        if user_data and isinstance(user_data, tuple):
            action, _ = user_data
        else:
            action = self._cmd_action_combo.currentText().split("  —  ")[0].strip()

        if not action or action == "(free text)":
            self._cmd_error_label.setText("Action required")
            return

        payload_text = self._cmd_payload_edit.toPlainText().strip() or "{}"
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
        user_data = self._cmd_action_combo.itemData(index)
        if user_data and isinstance(user_data, tuple):
            _, payload = user_data
            self._cmd_payload_edit.setText(json.dumps(payload, indent=2) if payload else "{}")

    # ------------------------------------------------------------------
    # Status badge
    # ------------------------------------------------------------------

    _STATUS_STYLES = {
        "STOPPED":    "background:#c0392b;color:white;border-radius:4px;padding:2px 8px;font-weight:bold;",
        "BOOK READY": "background:#e67e22;color:white;border-radius:4px;padding:2px 8px;font-weight:bold;",
        "RUNNING":    "background:#27ae60;color:white;border-radius:4px;padding:2px 8px;font-weight:bold;",
    }
    _STATUS_TEXT_COLOURS = {
        "RUNNING": "#27ae60", "BOOK READY": "#e67e22", "STOPPED": "#c0392b",
    }

    def _set_status(self, status: str) -> None:
        self._status_badge.setText(status)
        self._status_badge.setStyleSheet(self._STATUS_STYLES.get(status, self._STATUS_STYLES["STOPPED"]))
        if hasattr(self, "_control_status_label"):
            colour = self._STATUS_TEXT_COLOURS.get(status, "#c0392b")
            self._control_status_label.setText(status)
            self._control_status_label.setStyleSheet(f"color: {colour};")
