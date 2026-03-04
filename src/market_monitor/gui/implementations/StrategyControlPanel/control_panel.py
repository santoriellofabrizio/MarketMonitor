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
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QProcess, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPalette
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QScrollArea,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem,
    QInputDialog, QApplication,
)

from market_monitor.gui.implementations.StrategyControlPanel.redis_status_thread import RedisStatusThread

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SETTINGS_PATH = Path.home() / ".marketmonitor" / "control_panel_settings.json"

_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_LEVEL_RANK: Dict[str, int] = {l: i for i, l in enumerate(_LOG_LEVELS)}
_LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")
_LEVEL_COLOR: Dict[str, str] = {
    "DEBUG":    "#7f8c8d",
    "INFO":     "#d0d0d0",
    "WARNING":  "#e67e22",
    "ERROR":    "#e74c3c",
    "CRITICAL": "#c0392b",
}

_BTN_PRIMARY = (
    "QPushButton{background:#5b8dee;color:white;font-weight:bold;"
    "border-radius:5px;padding:5px 14px;}"
    "QPushButton:hover{background:#7aa3f5;}"
    "QPushButton:disabled{background:#3a3a5c;color:#888;}"
)
_BTN_SUCCESS = (
    "QPushButton{background:#27ae60;color:white;font-weight:bold;"
    "border-radius:5px;padding:5px 14px;}"
    "QPushButton:hover{background:#2ecc71;}"
)
_BTN_DANGER = (
    "QPushButton{background:#c0392b;color:white;font-weight:bold;"
    "border-radius:5px;padding:5px 14px;}"
    "QPushButton:hover{background:#e74c3c;}"
)
_BTN_NEUTRAL = (
    "QPushButton{background:#3a3a5c;color:#d0d0d0;font-weight:bold;"
    "border-radius:5px;padding:5px 14px;}"
    "QPushButton:hover{background:#4a4a6c;}"
)


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
        panel_config: Optional[dict] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.config_name = config_name
        self.commands_channel = commands_channel
        self.status_channel = status_channel
        self.lifecycle_channel = lifecycle_channel
        self.panel_config: dict = panel_config or {}
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
        self._status_threads: Dict[str, RedisStatusThread] = {}
        self._signals = _Signals()
        self._instances: List[StrategyInstance] = []
        self._all_configs: List[str] = []

        # Log state: per-instance buffers and tab widgets
        self._log_buffers: Dict[str, List[Tuple[str, str]]] = {"_all": []}
        self._log_tabs: Dict[str, QTextEdit] = {}
        self._log_filters: Dict[str, QComboBox] = {}

        # Load persisted settings
        self._settings = self._load_settings()

        self._build_redis_client(redis_config or {})
        self._setup_ui(initial_config)
        self._connect_signals()

        if self._lifecycle_thread:
            self._lifecycle_thread.start()

        # Restore previously saved configs (skip initial_config which is already added)
        for cfg in self._settings.get("configs", []):
            if cfg != initial_config:
                self._add_instance(cfg)

        # Restore saved log level filters
        for key, combo in self._log_filters.items():
            saved = self._settings.get("log_levels", {}).get(key, "DEBUG")
            idx = combo.findText(saved)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        self._update_status_bar()
        logger.info("StrategyControlPanel initialized")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_settings()
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
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> dict:
        try:
            if _SETTINGS_PATH.exists():
                return json.loads(_SETTINGS_PATH.read_text())
        except Exception:
            pass
        return {}

    def _save_settings(self) -> None:
        try:
            _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            geo = self.geometry()
            log_levels = {key: combo.currentText() for key, combo in self._log_filters.items()}
            data = {
                "version": "1.0",
                "window": {"x": geo.x(), "y": geo.y(), "width": geo.width(), "height": geo.height()},
                "configs": [inst.config_name for inst in self._instances],
                "log_levels": log_levels,
            }
            _SETTINGS_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not save settings: {e}")

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
            self._ensure_status_thread(self._default_status_channel)
            if self._lifecycle_channel:
                self._lifecycle_thread = RedisStatusThread(
                    self._redis_client, self._lifecycle_channel
                )
        except redis.RedisError as e:
            logger.warning(f"Redis not available: {e}. Commands will be disabled.")
            self._redis_client = None

    def _ensure_status_thread(self, channel: str) -> None:
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

    def _apply_dark_theme(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        app.setStyle("Fusion")
        palette = QPalette()
        bg       = QColor("#1e1e2e")
        bg_alt   = QColor("#28283e")
        bg_mid   = QColor("#2e2e4e")
        text     = QColor("#d0d0d0")
        text_dim = QColor("#888888")
        accent   = QColor("#5b8dee")
        border   = QColor("#3a3a5c")

        palette.setColor(QPalette.Window,          bg)
        palette.setColor(QPalette.WindowText,      text)
        palette.setColor(QPalette.Base,            bg_alt)
        palette.setColor(QPalette.AlternateBase,   bg_mid)
        palette.setColor(QPalette.ToolTipBase,     bg_mid)
        palette.setColor(QPalette.ToolTipText,     text)
        palette.setColor(QPalette.Text,            text)
        palette.setColor(QPalette.Button,          bg_mid)
        palette.setColor(QPalette.ButtonText,      text)
        palette.setColor(QPalette.BrightText,      QColor("white"))
        palette.setColor(QPalette.Link,            accent)
        palette.setColor(QPalette.Highlight,       accent)
        palette.setColor(QPalette.HighlightedText, QColor("white"))
        palette.setColor(QPalette.Disabled, QPalette.Text,       text_dim)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, text_dim)
        palette.setColor(QPalette.Mid,             border)
        palette.setColor(QPalette.Dark,            bg)
        app.setPalette(palette)

        app.setStyleSheet("""
            QGroupBox {
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 6px;
                font-weight: bold;
                color: #9999cc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a5c;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #28283e;
                color: #9090b0;
                padding: 6px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background: #1e1e2e;
                color: #d0d0d0;
                border-bottom: 2px solid #5b8dee;
            }
            QTabBar::tab:hover:!selected { background: #32325a; }
            QHeaderView::section {
                background: #28283e;
                color: #9090b0;
                padding: 4px 8px;
                border: none;
                border-right: 1px solid #3a3a5c;
                font-weight: bold;
            }
            QTableWidget { gridline-color: #3a3a5c; }
            QScrollBar:vertical {
                background: #1e1e2e; width: 10px; border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a5c; border-radius: 5px; min-height: 20px;
            }
            QComboBox {
                background: #28283e; color: #d0d0d0;
                border: 1px solid #3a3a5c; border-radius: 4px; padding: 3px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #28283e; color: #d0d0d0;
                selection-background-color: #5b8dee;
            }
            QTextEdit {
                background: #12121e; color: #d0d0d0;
                border: 1px solid #3a3a5c; border-radius: 4px;
            }
            QListWidget {
                background: #12121e; color: #d0d0d0;
                border: 1px solid #3a3a5c; border-radius: 4px;
            }
            QStatusBar { color: #888; font-size: 11px; }
            QLabel { color: #d0d0d0; }
        """)

    def _setup_ui(self, initial_config: Optional[str] = None) -> None:
        self._apply_dark_theme()

        saved_win = self._settings.get("window", {})
        x = saved_win.get("x", 200)
        y = saved_win.get("y", 200)
        w = saved_win.get("width", 1100)
        h = saved_win.get("height", 740)
        self.setGeometry(x, y, w, h)
        self.setMinimumSize(820, 560)
        self.setWindowTitle("Strategy Control Panel")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Header
        header = QHBoxLayout()
        title_lbl = QLabel("Strategy Control Panel")
        title_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        title_lbl.setStyleSheet("color: #9999ee;")
        self._redis_status_lbl = QLabel("● Redis: connecting…")
        self._redis_status_lbl.setStyleSheet("color: #e67e22; font-size: 11px;")
        save_btn = QPushButton("Save Settings")
        save_btn.setFixedWidth(110)
        save_btn.setStyleSheet(_BTN_NEUTRAL)
        save_btn.clicked.connect(self._save_settings)
        header.addWidget(title_lbl)
        header.addStretch()
        header.addWidget(self._redis_status_lbl)
        header.addSpacing(12)
        header.addWidget(save_btn)
        root.addLayout(header)

        self._tabs = QTabWidget()
        self._tabs.setFont(QFont("Arial", 10))
        root.addWidget(self._tabs)
        self._tabs.addTab(self._build_control_tab(initial_config), "Control")
        self._tabs.addTab(self._build_commands_tab(),               "Commands")
        self._tabs.addTab(self._build_events_tab(),                 "Events")
        self._tabs.addTab(self._build_logs_tab(),                   "Logs")

        # Status bar
        self._strategy_status_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self._strategy_status_lbl)
        self._update_redis_indicator()

    # --- Control tab ---

    def _build_control_tab(self, initial_config: Optional[str] = None) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        add_btn = QPushButton("+ Add Strategy")
        add_btn.setStyleSheet(_BTN_SUCCESS)
        add_btn.clicked.connect(self._on_add_strategy)
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet(_BTN_DANGER)
        remove_btn.clicked.connect(self._on_remove_strategy)
        toolbar.addWidget(add_btn)
        toolbar.addWidget(remove_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._strategy_table = QTableWidget(0, 5)
        self._strategy_table.setHorizontalHeaderLabels(
            ["Config", "Status", "PID", "Start", "Stop"]
        )
        self._strategy_table.setAlternatingRowColors(True)
        hdr = self._strategy_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col, w_px in [(1, 90), (2, 80), (3, 80), (4, 80)]:
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self._strategy_table.setColumnWidth(col, w_px)
        self._strategy_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._strategy_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._strategy_table.verticalHeader().setVisible(False)
        self._strategy_table.setShowGrid(False)
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
            inst = self._instances[row]
            inst.terminate()
            self._remove_log_tab(inst.config_name)
            self._instances.pop(row)
            self._strategy_table.removeRow(row)
            self._update_target_combo()
            self._rebuild_quick_commands()
            self._update_status_bar()

    def _add_instance(self, config_name: str) -> None:
        # Load panel_config from YAML for this specific instance
        panel_cfg: dict = {}
        try:
            from market_monitor.utils.config_helpers import find_config, load_config
            cfg = load_config(find_config(config_name))
            panel_cfg = cfg.get("control_panel", {})
        except Exception:
            pass

        channels = self._channels_for_config(config_name)
        inst = StrategyInstance(
            config_name=config_name,
            commands_channel=channels["commands_channel"],
            status_channel=channels["status_channel"],
            lifecycle_channel=channels["lifecycle_channel"],
            panel_config=panel_cfg,
            parent=self,
        )
        inst.status_changed.connect(self._on_instance_status_changed)
        inst.output_received.connect(self._on_instance_output)
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
        start_btn.setStyleSheet(_BTN_SUCCESS)
        start_btn.clicked.connect(lambda _checked, i=inst: i.start())
        stop_btn = QPushButton("Stop")
        stop_btn.setStyleSheet(_BTN_DANGER)
        stop_btn.clicked.connect(lambda _checked, i=inst: i.stop(self._redis_client))
        self._strategy_table.setCellWidget(row, 3, start_btn)
        self._strategy_table.setCellWidget(row, 4, stop_btn)

        self._update_target_combo()
        self._rebuild_quick_commands()
        self._add_log_tab(inst)
        self._update_status_bar()

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
                "commands_channel":  cl.get("channel",           defaults["commands_channel"]),
                "status_channel":    cl.get("status_channel",    defaults["status_channel"]),
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
        self._update_status_bar()

    # --- Commands tab ---

    def _build_commands_tab(self) -> QWidget:
        w = QWidget()
        self._commands_tab_layout = QVBoxLayout(w)
        self._commands_tab_layout.setSpacing(8)

        # Quick commands container (per-strategy groups in a scroll area)
        self._quick_scroll = QScrollArea()
        self._quick_scroll.setWidgetResizable(True)
        self._quick_scroll.setMaximumHeight(200)
        self._quick_scroll.setFrameShape(QScrollArea.NoFrame)
        self._quick_inner = QWidget()
        self._quick_inner_layout = QVBoxLayout(self._quick_inner)
        self._quick_inner_layout.setSpacing(4)
        self._quick_inner_layout.addStretch()
        self._quick_scroll.setWidget(self._quick_inner)
        self._commands_tab_layout.addWidget(self._quick_scroll)

        layout = self._commands_tab_layout

        # Target strategy selector
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target (free-form send):"))
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(260)
        self._target_combo.addItem("(all / default channel)")
        target_row.addWidget(self._target_combo)
        target_row.addStretch()
        layout.addLayout(target_row)

        # Send Command form
        send_grp = QGroupBox("Send Command")
        send_layout = QVBoxLayout(send_grp)
        send_layout.setSpacing(6)

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
        self._cmd_payload_edit.setMaximumHeight(90)
        self._cmd_payload_edit.setFont(QFont("Courier New", 9))
        send_layout.addWidget(self._cmd_payload_edit)

        btn_row = QHBoxLayout()
        self._send_btn = QPushButton("Send")
        self._send_btn.setStyleSheet(_BTN_PRIMARY)
        self._send_btn.setFixedWidth(100)
        self._send_btn.setEnabled(self._redis_client is not None)
        self._send_btn.clicked.connect(self._on_send_command)
        self._cmd_error_label = QLabel("")
        self._cmd_error_label.setStyleSheet("color: #e74c3c;")
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
        self._response_table.setAlternatingRowColors(True)
        self._response_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._response_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._response_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._response_table.setShowGrid(False)
        layout.addWidget(self._response_table)
        return w

    def _rebuild_quick_commands(self) -> None:
        """Rebuild the per-strategy quick-command groups from current instances."""
        # Remove all widgets except the trailing stretch
        while self._quick_inner_layout.count() > 1:
            item = self._quick_inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for inst in self._instances:
            commands = inst.panel_config.get("commands", [])
            if not commands:
                continue
            grp = QGroupBox(inst.config_name)
            grp_layout = QHBoxLayout(grp)
            grp_layout.setAlignment(Qt.AlignLeft)
            grp_layout.setSpacing(6)
            channel = inst.commands_channel
            for cmd in commands:
                label   = cmd.get("label", cmd.get("action", "?"))
                action  = cmd.get("action", "")
                payload = dict(cmd.get("payload", {}))
                desc    = cmd.get("description", "")
                btn = QPushButton(label)
                btn.setFixedHeight(30)
                btn.setStyleSheet(_BTN_PRIMARY)
                if desc:
                    btn.setToolTip(desc)
                btn.clicked.connect(
                    lambda _c, a=action, p=payload, ch=channel: self._send_quick_command(a, p, ch)
                )
                grp_layout.addWidget(btn)
            grp_layout.addStretch()
            # Insert before the trailing stretch
            self._quick_inner_layout.insertWidget(
                self._quick_inner_layout.count() - 1, grp
            )

        # Show/hide scroll area based on whether there's anything
        has_cmds = any(inst.panel_config.get("commands") for inst in self._instances)
        self._quick_scroll.setVisible(has_cmds)

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
        layout.setSpacing(6)

        btn_row = QHBoxLayout()
        lbl = QLabel("Lifecycle Events")
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        btn_row.addWidget(lbl)
        btn_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(80)
        clear_btn.setStyleSheet(_BTN_NEUTRAL)
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
        layout.setContentsMargins(4, 4, 4, 4)

        self._log_tab_widget = QTabWidget()
        self._log_tab_widget.setFont(QFont("Arial", 9))
        layout.addWidget(self._log_tab_widget)

        # "All" tab (always present)
        all_widget, all_edit, all_combo = self._make_log_tab_widget("_all")
        self._log_all = all_edit
        self._log_tab_widget.addTab(all_widget, "All")
        return w

    def _make_log_tab_widget(self, key: str) -> Tuple[QWidget, QTextEdit, QComboBox]:
        """Create the content widget for a log tab. Returns (widget, text_edit, filter_combo)."""
        tab_w = QWidget()
        tab_layout = QVBoxLayout(tab_w)
        tab_layout.setContentsMargins(4, 4, 4, 4)
        tab_layout.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Min level:"))
        combo = QComboBox()
        combo.setFixedWidth(110)
        for lvl in _LOG_LEVELS:
            combo.addItem(lvl)
        self._log_filters[key] = combo

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(70)
        clear_btn.setStyleSheet(_BTN_NEUTRAL)

        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setFont(QFont("Courier New", 9))
        edit.document().setMaximumBlockCount(5000)

        combo.currentIndexChanged.connect(
            lambda _idx, k=key, c=combo, e=edit: self._on_log_filter_changed(k, c, e)
        )
        clear_btn.clicked.connect(lambda: (edit.clear(), self._log_buffers.__setitem__(key, [])))

        ctrl.addWidget(combo)
        ctrl.addSpacing(10)
        ctrl.addWidget(clear_btn)
        ctrl.addStretch()
        tab_layout.addLayout(ctrl)
        tab_layout.addWidget(edit)
        return tab_w, edit, combo

    def _add_log_tab(self, inst: StrategyInstance) -> None:
        self._log_buffers[inst.config_name] = []
        tab_w, edit, _combo = self._make_log_tab_widget(inst.config_name)
        self._log_tabs[inst.config_name] = edit
        self._log_tab_widget.addTab(tab_w, inst.config_name)

    def _remove_log_tab(self, config_name: str) -> None:
        # Find the tab index
        for i in range(self._log_tab_widget.count()):
            if self._log_tab_widget.tabText(i) == config_name:
                self._log_tab_widget.removeTab(i)
                break
        self._log_tabs.pop(config_name, None)
        self._log_buffers.pop(config_name, None)
        self._log_filters.pop(config_name, None)

    # --- Log helpers ---

    @staticmethod
    def _detect_level(line: str) -> str:
        m = _LEVEL_RE.search(line)
        return m.group(1) if m else "DEBUG"

    @staticmethod
    def _format_log_html(level: str, text: str) -> str:
        color = _LEVEL_COLOR.get(level, _LEVEL_COLOR["INFO"])
        escaped = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .rstrip("\n")
        )
        return f'<span style="color:{color};">{escaped}</span>'

    def _append_html_line(self, edit: QTextEdit, html: str) -> None:
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not edit.toPlainText():
            cursor.insertHtml(html)
        else:
            cursor.insertHtml("<br>" + html)
        edit.setTextCursor(cursor)
        edit.ensureCursorVisible()

    def _on_log_filter_changed(self, key: str, combo: QComboBox, edit: QTextEdit) -> None:
        min_rank = _LEVEL_RANK.get(combo.currentText(), 0)
        edit.clear()
        buf = self._log_buffers.get(key, [])
        first = True
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        for level, line in buf:
            if _LEVEL_RANK.get(level, 0) >= min_rank:
                html = self._format_log_html(level, line)
                if first:
                    cursor.insertHtml(html)
                    first = False
                else:
                    cursor.insertHtml("<br>" + html)
        edit.setTextCursor(cursor)
        edit.ensureCursorVisible()

    def _on_instance_output(self, inst: StrategyInstance, text: str) -> None:
        for line in text.splitlines():
            if not line.strip():
                continue
            level = self._detect_level(line)
            prefixed = f"[{inst.config_name}] {line}"

            # Append to buffers
            self._log_buffers.setdefault(inst.config_name, []).append((level, f"[{inst.config_name}] {line}"))
            self._log_buffers["_all"].append((level, prefixed))

            # Check filters and append to visible edits
            min_rank_inst = _LEVEL_RANK.get(
                self._log_filters.get(inst.config_name, QComboBox()).currentText() if inst.config_name in self._log_filters else "DEBUG", 0
            )
            min_rank_all = _LEVEL_RANK.get(
                self._log_filters.get("_all", QComboBox()).currentText() if "_all" in self._log_filters else "DEBUG", 0
            )
            if _LEVEL_RANK.get(level, 0) >= min_rank_inst and inst.config_name in self._log_tabs:
                self._append_html_line(
                    self._log_tabs[inst.config_name],
                    self._format_log_html(level, f"[{inst.config_name}] {line}"),
                )
            if _LEVEL_RANK.get(level, 0) >= min_rank_all:
                self._append_html_line(
                    self._log_all,
                    self._format_log_html(level, prefixed),
                )

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_redis_indicator(self) -> None:
        if self._redis_client:
            self._redis_status_lbl.setText("● Redis: connected")
            self._redis_status_lbl.setStyleSheet("color: #27ae60; font-size: 11px;")
        else:
            self._redis_status_lbl.setText("● Redis: disconnected")
            self._redis_status_lbl.setStyleSheet("color: #e74c3c; font-size: 11px;")

    def _update_status_bar(self) -> None:
        total   = len(self._instances)
        running = sum(1 for i in self._instances if i.status == "RUNNING")
        self._strategy_status_lbl.setText(f"Strategies: {running} running / {total} loaded")

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

    def _send_quick_command(self, action: str, payload: dict, channel: str) -> None:
        if not self._redis_client:
            return
        try:
            msg = dict(payload)
            msg["action"] = action
            self._redis_client.publish(channel, json.dumps(msg))
            logger.info(f"Quick command sent on {channel}: {msg}")
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
                cell.setForeground(QColor("#e74c3c"))
            self._response_table.setItem(row, col, cell)
        self._response_table.scrollToBottom()

    def _on_redis_connection_error(self, error: str) -> None:
        self._redis_client = None
        self._update_redis_indicator()
        if hasattr(self, "_cmd_error_label"):
            self._cmd_error_label.setText(f"Redis: {error}")
        if hasattr(self, "_send_btn"):
            self._send_btn.setEnabled(False)
