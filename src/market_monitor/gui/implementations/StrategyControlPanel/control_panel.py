"""
StrategyControlPanel — standalone PyQt5 control panel for MarketMonitor strategies.

Fully decoupled from strategy code. Communicates exclusively via Redis.
Can optionally launch strategies as subprocesses.

Usage:
    run-control-panel [--host HOST] [--port PORT] [--config CONFIG_NAME]
                      [--lifecycle-channel engine:lifecycle]

Or connect to already-running strategies:
    run-control-panel   (panel shows up; strategy running elsewhere)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QProcess, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem,
    QInputDialog,
)

from market_monitor.gui.implementations.StrategyControlPanel.redis_status_thread import RedisStatusThread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qt signal bridge
# ---------------------------------------------------------------------------

class _Signals(QObject):
    lifecycle_event = pyqtSignal(str, object)   # (event_name, data)


# ---------------------------------------------------------------------------
# StrategyInstance — manages a single strategy subprocess
# ---------------------------------------------------------------------------

class StrategyInstance(QObject):
    """Wraps a single strategy: its QProcess and channel addresses."""

    status_changed = pyqtSignal(object, str)   # (self, new_status)
    output_received = pyqtSignal(object, str)  # (self, text)

    def __init__(
        self,
        config_name: str,
        commands_channel: str = "engine:commands",
        status_channel: str = "engine:status",
        lifecycle_channel: str = "engine:lifecycle",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.config_name = config_name
        self.commands_channel = commands_channel
        self.status_channel = status_channel
        self.lifecycle_channel = lifecycle_channel
        self.status = "STOPPED"
        self.pid: Optional[int] = None

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)

    def start(self) -> bool:
        if self._process.state() != QProcess.NotRunning:
            return False
        self._process.start("run-strategy", [self.config_name])
        if self._process.waitForStarted(3000):
            self.pid = self._process.processId()
            self.status = "RUNNING"
            self.status_changed.emit(self, "RUNNING")
            return True
        self.status = "ERROR"
        self.status_changed.emit(self, "ERROR")
        return False

    def stop(self, redis_client: Optional[redis.StrictRedis] = None) -> None:
        if redis_client:
            try:
                redis_client.publish(self.commands_channel, json.dumps({"action": "stop"}))
            except Exception:
                pass
        QTimer.singleShot(3000, self._kill_if_running)

    def terminate(self) -> None:
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
        self.status = "STOPPED"
        self.pid = None

    def _kill_if_running(self) -> None:
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def _on_output(self) -> None:
        data = self._process.readAllStandardOutput().data().decode(errors="replace")
        self.output_received.emit(self, data)

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self.pid = None
        self.status = "STOPPED"
        self.status_changed.emit(self, "STOPPED")
        self.output_received.emit(self, f"\n[Process exited with code {exit_code}]\n")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class StrategyControlPanel(QMainWindow):
    """
    Standalone PyQt5 control panel.

    Does NOT depend on StrategyUIAsync, Builder, or any strategy class.
    All communication with strategies happens via Redis pub/sub:

        engine:commands   → sends user commands
        engine:status     → receives command responses
        engine:lifecycle  → receives lifecycle events

    The panel can also launch and kill strategy subprocesses via QProcess.
    Multiple strategies can be managed simultaneously.
    """

    def __init__(
        self,
        redis_config: Optional[dict] = None,
        commands_channel: str = "engine:commands",
        status_channel: str = "engine:status",
        lifecycle_channel: Optional[str] = None,
        initial_config: Optional[str] = None,
        panel_config: Optional[dict] = None,
        **kwargs,
    ):
        QMainWindow.__init__(self)

        self._default_commands_channel = commands_channel
        self._default_status_channel = status_channel
        self._lifecycle_channel = lifecycle_channel
        self._redis_client: Optional[redis.StrictRedis] = None
        self._lifecycle_thread: Optional[RedisStatusThread] = None
        # One RedisStatusThread per unique status channel
        self._status_threads: Dict[str, RedisStatusThread] = {}
        self._signals = _Signals()
        self._panel_config = panel_config or {}
        self._instances: List[StrategyInstance] = []
        self._all_configs: List[str] = []

        self._build_redis_client(redis_config or {})
        self._setup_ui(initial_config)
        self._connect_signals()

        if self._lifecycle_thread:
            self._lifecycle_thread.start()

        logger.info("StrategyControlPanel initialized")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._lifecycle_thread and self._lifecycle_thread.isRunning():
            self._lifecycle_thread.stop()
            self._lifecycle_thread.wait(2000)
        for thread in self._status_threads.values():
            if thread.isRunning():
                thread.stop()
                thread.wait(2000)
        for inst in self._instances:
            inst.terminate()
        super().closeEvent(event)

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
            # Always start a thread on the default status channel
            self._ensure_status_thread(self._default_status_channel)
            if self._lifecycle_channel:
                self._lifecycle_thread = RedisStatusThread(
                    self._redis_client, self._lifecycle_channel
                )
        except redis.RedisError as e:
            logger.warning(f"Redis not available: {e}. Commands will be disabled.")
            self._redis_client = None

    def _ensure_status_thread(self, channel: str) -> None:
        """Start a RedisStatusThread for `channel` if not already running."""
        if channel in self._status_threads or not self._redis_client:
            return
        thread = RedisStatusThread(self._redis_client, channel)
        thread.status_received.connect(self._on_status_received)
        thread.connection_error.connect(self._on_redis_connection_error)
        thread.start()
        self._status_threads[channel] = thread

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self, initial_config: Optional[str] = None) -> None:
        self.setWindowTitle("Strategy Control Panel")
        self.setGeometry(200, 200, 1050, 720)
        self.setMinimumSize(800, 540)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        title_lbl = QLabel("Strategy Control Panel")
        title_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        header.addWidget(title_lbl)
        header.addStretch()
        root.addLayout(header)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)
        self._tabs.addTab(self._build_control_tab(initial_config), "Control")
        self._tabs.addTab(self._build_commands_tab(),               "Commands")
        self._tabs.addTab(self._build_events_tab(),                 "Events")
        self._tabs.addTab(self._build_logs_tab(),                   "Logs")

    # --- Control tab ---

    def _build_control_tab(self, initial_config: Optional[str] = None) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        toolbar = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.setFixedWidth(80)
        add_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;"
            "border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#2ecc71;}"
        )
        add_btn.clicked.connect(self._on_add_strategy)
        remove_btn = QPushButton("Remove")
        remove_btn.setFixedWidth(80)
        remove_btn.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;font-weight:bold;"
            "border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#e74c3c;}"
        )
        remove_btn.clicked.connect(self._on_remove_strategy)
        toolbar.addWidget(add_btn)
        toolbar.addWidget(remove_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._strategy_table = QTableWidget(0, 5)
        self._strategy_table.setHorizontalHeaderLabels(
            ["Config", "Status", "PID", "Start", "Stop"]
        )
        hdr = self._strategy_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.Fixed)
        self._strategy_table.setColumnWidth(1, 90)
        self._strategy_table.setColumnWidth(2, 80)
        self._strategy_table.setColumnWidth(3, 70)
        self._strategy_table.setColumnWidth(4, 70)
        self._strategy_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._strategy_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._strategy_table.verticalHeader().setVisible(False)
        layout.addWidget(self._strategy_table)

        try:
            from market_monitor.utils.config_helpers import find_all_configs
            self._all_configs = list(find_all_configs())
        except Exception as e:
            logger.debug(f"Could not load config list: {e}")

        if initial_config:
            self._add_instance(initial_config)

        return w

    def _on_add_strategy(self) -> None:
        if self._all_configs:
            name, ok = QInputDialog.getItem(
                self, "Add Strategy", "Select config:", self._all_configs, 0, True
            )
        else:
            name, ok = QInputDialog.getText(self, "Add Strategy", "Config name:")
        if ok and name.strip():
            self._add_instance(name.strip())

    def _on_remove_strategy(self) -> None:
        rows = self._strategy_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row < len(self._instances):
            self._instances[row].terminate()
            self._instances.pop(row)
            self._strategy_table.removeRow(row)
            self._update_target_combo()

    def _add_instance(self, config_name: str) -> None:
        # If no quick-commands loaded yet, pull them from this config's YAML
        if not self._panel_config.get("commands"):
            try:
                from market_monitor.utils.config_helpers import find_config, load_config
                cfg = load_config(find_config(config_name))
                panel_cfg = cfg.get("control_panel", {})
                if panel_cfg.get("commands"):
                    self._panel_config = panel_cfg
                    self._rebuild_quick_commands(panel_cfg["commands"])
            except Exception:
                pass

        channels = self._channels_for_config(config_name)
        inst = StrategyInstance(
            config_name=config_name,
            commands_channel=channels["commands_channel"],
            status_channel=channels["status_channel"],
            lifecycle_channel=channels["lifecycle_channel"],
            parent=self,
        )
        inst.status_changed.connect(self._on_instance_status_changed)
        inst.output_received.connect(self._on_instance_output)
        # Ensure a status thread covers this instance's channel
        self._ensure_status_thread(channels["status_channel"])
        self._instances.append(inst)

        row = self._strategy_table.rowCount()
        self._strategy_table.insertRow(row)
        self._strategy_table.setItem(row, 0, QTableWidgetItem(config_name))
        status_item = QTableWidgetItem("STOPPED")
        status_item.setForeground(QColor("#c0392b"))
        self._strategy_table.setItem(row, 1, status_item)
        self._strategy_table.setItem(row, 2, QTableWidgetItem("—"))

        start_btn = QPushButton("Start")
        start_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;border-radius:3px;}"
            "QPushButton:hover{background:#2ecc71;}"
        )
        start_btn.clicked.connect(lambda _checked, i=inst: i.start())

        stop_btn = QPushButton("Stop")
        stop_btn.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;font-weight:bold;border-radius:3px;}"
            "QPushButton:hover{background:#e74c3c;}"
        )
        stop_btn.clicked.connect(lambda _checked, i=inst: i.stop(self._redis_client))

        self._strategy_table.setCellWidget(row, 3, start_btn)
        self._strategy_table.setCellWidget(row, 4, stop_btn)
        self._update_target_combo()

    def _channels_for_config(self, config_name: str) -> dict:
        defaults = {
            "commands_channel":  self._default_commands_channel,
            "status_channel":    self._default_status_channel,
            "lifecycle_channel": self._lifecycle_channel or "engine:lifecycle",
        }
        try:
            from market_monitor.utils.config_helpers import find_config, load_config
            config = load_config(find_config(config_name))
            cl = config.get("tasks", {}).get("command_listener", {})
            return {
                "commands_channel":  cl.get("channel",          defaults["commands_channel"]),
                "status_channel":    cl.get("status_channel",   defaults["status_channel"]),
                "lifecycle_channel": cl.get("lifecycle_channel", defaults["lifecycle_channel"]),
            }
        except Exception:
            return defaults

    def _on_instance_status_changed(self, inst: StrategyInstance, status: str) -> None:
        row = self._instances.index(inst) if inst in self._instances else -1
        if row < 0:
            return
        colour = {"RUNNING": "#27ae60", "ERROR": "#e67e22"}.get(status, "#c0392b")
        status_item = QTableWidgetItem(status)
        status_item.setForeground(QColor(colour))
        self._strategy_table.setItem(row, 1, status_item)
        self._strategy_table.setItem(row, 2, QTableWidgetItem(str(inst.pid) if inst.pid else "—"))

    def _on_instance_output(self, inst: StrategyInstance, text: str) -> None:
        prefix = f"[{inst.config_name}] "
        lines = text.splitlines(keepends=True)
        prefixed = "".join(prefix + ln if ln.strip() else ln for ln in lines)
        cursor = self._log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_edit.setTextCursor(cursor)
        self._log_edit.insertPlainText(prefixed)
        self._log_edit.ensureCursorVisible()

    # --- Commands tab ---

    def _build_commands_tab(self) -> QWidget:
        w = QWidget()
        self._commands_tab_layout = QVBoxLayout(w)
        self._quick_commands_widget = None

        self._rebuild_quick_commands(self._panel_config.get("commands", []))

        layout = self._commands_tab_layout

        # Target strategy selector
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target Strategy:"))
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(250)
        self._target_combo.addItem("(all / default channel)")
        target_row.addWidget(self._target_combo)
        target_row.addStretch()
        layout.addLayout(target_row)

        # Send Command form
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

    def _rebuild_quick_commands(self, commands: list) -> None:
        """Replace the Quick Commands group with buttons built from `commands`."""
        layout = self._commands_tab_layout
        if self._quick_commands_widget is not None:
            layout.removeWidget(self._quick_commands_widget)
            self._quick_commands_widget.deleteLater()
            self._quick_commands_widget = None
        if not commands:
            return
        quick_grp = QGroupBox("Quick Commands")
        quick_layout = QHBoxLayout(quick_grp)
        quick_layout.setAlignment(Qt.AlignLeft)
        for cmd in commands:
            label = cmd.get("label", cmd.get("action", "?"))
            action = cmd.get("action", "")
            payload = dict(cmd.get("payload", {}))
            description = cmd.get("description", "")
            btn = QPushButton(label)
            btn.setFixedHeight(32)
            if description:
                btn.setToolTip(description)
            btn.clicked.connect(
                lambda _checked, a=action, p=payload: self._send_quick_command(a, p)
            )
            quick_layout.addWidget(btn)
        quick_layout.addStretch()
        layout.insertWidget(0, quick_grp)
        self._quick_commands_widget = quick_grp

    def _update_target_combo(self) -> None:
        if not hasattr(self, "_target_combo"):
            return
        current = self._target_combo.currentText()
        self._target_combo.clear()
        self._target_combo.addItem("(all / default channel)")
        for inst in self._instances:
            self._target_combo.addItem(inst.config_name)
        idx = self._target_combo.findText(current)
        if idx >= 0:
            self._target_combo.setCurrentIndex(idx)

    def _resolve_commands_channel(self) -> str:
        if not hasattr(self, "_target_combo"):
            return self._default_commands_channel
        idx = self._target_combo.currentIndex()
        if idx > 0 and (idx - 1) < len(self._instances):
            return self._instances[idx - 1].commands_channel
        return self._default_commands_channel

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
        if self._lifecycle_thread:
            self._lifecycle_thread.status_received.connect(self._on_lifecycle_redis_message)
            self._lifecycle_thread.connection_error.connect(self._on_redis_connection_error)

    # ------------------------------------------------------------------
    # Slots — Commands
    # ------------------------------------------------------------------

    def _send_quick_command(self, action: str, payload: dict) -> None:
        if not self._redis_client:
            return
        try:
            msg = dict(payload)
            msg["action"] = action
            channel = self._resolve_commands_channel()
            self._redis_client.publish(channel, json.dumps(msg))
            logger.info(f"Quick command sent: {msg}")
        except redis.RedisError as e:
            logger.error(f"Redis publish error: {e}")

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
            channel = self._resolve_commands_channel()
            self._redis_client.publish(channel, json.dumps(payload))
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
        if hasattr(self, "_cmd_error_label"):
            self._cmd_error_label.setText(f"Redis: {error}")
        if hasattr(self, "_send_btn"):
            self._send_btn.setEnabled(False)
