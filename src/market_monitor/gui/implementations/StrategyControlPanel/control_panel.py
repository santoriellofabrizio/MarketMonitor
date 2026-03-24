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

import atexit
import json
import logging
import re
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import redis
from PyQt5.QtCore import Qt, QEvent, pyqtSignal, QObject, QProcess, QTimer, QUrl
from PyQt5.QtGui import (
    QFont, QColor, QTextCursor, QPalette, QIcon, QPixmap, QPainter,
    QSyntaxHighlighter, QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QScrollArea,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QGroupBox, QDoubleSpinBox, QLineEdit,
    QApplication, QSystemTrayIcon, QMenu, QAction,
    QFileDialog, QCheckBox, QSplitter, QListWidget, QListWidgetItem,
    QMessageBox,
)
from PyQt5.QtGui import QDesktopServices

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

# Row background colours for strategy status
_ROW_BG = {
    "RUNNING":    QColor("#1a3a1a"),
    "ERROR":      QColor("#3a1a1a"),
    "RESTARTING": QColor("#3a2a0a"),
}

# Lifecycle / stats log parsers
_LC_RE    = re.compile(r"\[LIFECYCLE\]\s+(\S+)(?:\s+count=(\d+))?")
_STATS_RE = re.compile(r"\[STATS\]\s+(\w+)=([\d.]+)s")

def _fmt_uptime(seconds: float) -> str:
    """Format elapsed seconds as '2h 15m' / '45s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m {s % 60}s"
    h = m // 60
    return f"{h}h {m % 60}m"


# ---------------------------------------------------------------------------
# _YamlHighlighter — minimal YAML syntax colouring for the config editor
# ---------------------------------------------------------------------------

class _YamlHighlighter(QSyntaxHighlighter):
    """Light YAML syntax highlighter: keys, values, comments, booleans, numbers."""

    def __init__(self, document):
        super().__init__(document)

        def _fmt(hex_color: str, italic: bool = False, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(hex_color))
            if italic:
                f.setFontItalic(True)
            if bold:
                f.setFontWeight(QFont.Bold)
            return f

        self._comment = _fmt("#7f8c8d", italic=True)
        self._key     = _fmt("#7ec8e3", bold=True)
        self._string  = _fmt("#98c379")
        self._keyword = _fmt("#e5c07b")   # true/false/null/yes/no
        self._number  = _fmt("#d19a66")
        self._anchor  = _fmt("#c678dd")   # &anchor / *ref / --- / ...

        self._rules = [
            # inline comment (after content)
            (re.compile(r"(?<!['\"])\s#.*$"), self._comment),
            # quoted strings
            (re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"'), self._string),
            (re.compile(r"'[^']*'"),                   self._string),
            # YAML keywords
            (re.compile(r"\b(true|false|null|yes|no|on|off)\b", re.IGNORECASE), self._keyword),
            # numbers (int, float, negative)
            (re.compile(r"(?<![:\w])-?\b\d+\.?\d*\b"), self._number),
            # anchors/aliases/directives
            (re.compile(r"[&*][A-Za-z_]\w*|^---$|^\.\.\.$"), self._anchor),
        ]

    def highlightBlock(self, text: str) -> None:
        # Full-line comment
        if re.match(r"^\s*#", text):
            self.setFormat(0, len(text), self._comment)
            return
        # Top-level or nested key  (word chars before the first colon)
        m = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_.]*)\s*:", text)
        if m:
            self.setFormat(m.start(2), len(m.group(2)), self._key)
        # Other rules
        for pattern, fmt in self._rules:
            for mo in pattern.finditer(text):
                self.setFormat(mo.start(), mo.end() - mo.start(), fmt)


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
        self.auto_restart: bool = False
        self.restart_count: int = 0
        self.started_at: Optional[float] = None   # set by panel when [LIFECYCLE] started arrives

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
        self.started_at = None
        self.output_received.emit(self, f"\n[Process exited with code {exit_code}]\n")
        if self.auto_restart and exit_code != 0:
            self.restart_count += 1
            self.status = "RESTARTING"
            self.status_changed.emit(self, "RESTARTING")
            QTimer.singleShot(3000, self.start)
        else:
            self.status = "STOPPED"
            self.status_changed.emit(self, "STOPPED")


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
        self._status_threads: Dict[str, RedisStatusThread] = {}
        self._instances: List[StrategyInstance] = []
        self._all_configs: List[str] = []

        # Log state: per-instance buffers and tab widgets
        self._log_buffers: Dict[str, List[Tuple[str, str]]] = {"_all": []}
        self._log_tabs: Dict[str, QTextEdit] = {}
        self._log_filters: Dict[str, QComboBox] = {}
        self._log_searches: Dict[str, QLineEdit] = {}

        # Per-instance lifecycle statistics (populated by log parsing)
        self._lifecycle_stats: Dict[str, dict] = {}

        # Load persisted settings
        self._settings = self._load_settings()

        # Register atexit cleanup so child processes are killed even on crash
        _self_ref = weakref.ref(self)
        def _atexit_cleanup():
            panel = _self_ref()
            if panel is not None:
                for inst in list(panel._instances):
                    inst.terminate()
        atexit.register(_atexit_cleanup)

        self._build_redis_client(redis_config or {})
        self._setup_ui(initial_config)

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
        logger.info(
            f"closeEvent — windowState=0x{int(self.windowState()):02x} "
            f"isVisible={self.isVisible()}"
        )
        self._save_settings()
        if hasattr(self, "_tray"):
            self._tray.hide()
        for thread in self._status_threads.values():
            if thread.isRunning():
                thread.stop()
                thread.wait(2000)
        for inst in self._instances:
            inst.terminate()
        super().closeEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.WindowStateChange:
            state        = int(self.windowState())
            minimized    = bool(self.windowState() & Qt.WindowMinimized)
            tray_enabled = hasattr(self, "_tray_toggle") and self._tray_toggle.isChecked()
            tray_available = QSystemTrayIcon.isSystemTrayAvailable()
            tray_visible = hasattr(self, "_tray") and self._tray.isVisible()
            logger.info(
                f"WindowStateChange — state=0x{state:02x} minimized={minimized} "
                f"minimize_to_tray={tray_enabled} tray_available={tray_available} "
                f"tray_visible={tray_visible}"
            )
            # Only hide if the user explicitly enabled it AND the tray actually works.
            # If any guard fails, the window stays minimized (normal taskbar behaviour)
            # so the user can always restore it.
            if minimized and tray_enabled and tray_available and tray_visible:
                logger.info("Hiding window to system tray")
                QTimer.singleShot(0, self.hide)
            elif minimized and tray_enabled and not (tray_available and tray_visible):
                logger.warning(
                    "minimize_to_tray is on but system tray is not usable "
                    f"(available={tray_available} visible={tray_visible}) — "
                    "leaving window minimized in taskbar instead"
                )
        super().changeEvent(event)

    def _restore_from_tray(self) -> None:
        logger.info("_restore_from_tray called")
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _on_tray_activated(self, reason) -> None:
        logger.info(f"Tray icon activated — reason={reason}")
        if reason == QSystemTrayIcon.DoubleClick:
            self._restore_from_tray()

    @staticmethod
    def _make_tray_icon(running: bool) -> QIcon:
        """Generate a simple colored circle icon for the tray."""
        px = QPixmap(16, 16)
        px.fill(Qt.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor("#27ae60") if running else QColor("#555577"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, 14, 14)
        p.end()
        return QIcon(px)

    def _update_tray_icon(self) -> None:
        running = any(i.status == "RUNNING" for i in self._instances)
        if hasattr(self, "_tray"):
            self._tray.setIcon(self._make_tray_icon(running))
            count = sum(1 for i in self._instances if i.status == "RUNNING")
            self._tray.setToolTip(
                f"Strategy Control Panel — {count} running" if count else "Strategy Control Panel"
            )

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
            # Use normalGeometry() so saved dimensions are the restored (non-maximized) size
            geo = self.normalGeometry()
            log_levels = {key: combo.currentText() for key, combo in self._log_filters.items()}
            data = {
                "version": "1.0",
                "window": {
                    "x": geo.x(), "y": geo.y(),
                    "width": geo.width(), "height": geo.height(),
                    "maximized": self.isMaximized(),
                },
                "configs": [inst.config_name for inst in self._instances],
                "log_levels": log_levels,
                "minimize_to_tray": (
                    self._tray_toggle.isChecked()
                    if hasattr(self, "_tray_toggle") else True
                ),
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

        # Clamp geometry to available screen so window can't be stuck off-screen
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            x = max(avail.x(), min(x, avail.x() + avail.width() - 100))
            y = max(avail.y(), min(y, avail.y() + avail.height() - 100))
            w = min(w, avail.width())
            h = min(h, avail.height())

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
        self._tray_toggle = QCheckBox("Minimize to tray")
        self._tray_toggle.setChecked(self._settings.get("minimize_to_tray", False))
        self._tray_toggle.setStyleSheet("color:#9090b0; font-size:11px;")
        header.addWidget(title_lbl)
        header.addStretch()
        header.addWidget(self._tray_toggle)
        header.addSpacing(12)
        header.addWidget(self._redis_status_lbl)
        header.addSpacing(12)
        header.addWidget(save_btn)
        root.addLayout(header)

        self._tabs = QTabWidget()
        self._tabs.setFont(QFont("Arial", 10))
        root.addWidget(self._tabs)
        self._tabs.addTab(self._build_control_tab(initial_config), "Control")
        self._tabs.addTab(self._build_commands_tab(),               "Commands")
        self._tabs.addTab(self._build_logs_tab(),                   "Logs")
        self._tabs.addTab(self._build_config_tab(),                 "⚙ Config")

        # Status bar
        self._strategy_status_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self._strategy_status_lbl)
        self._update_redis_indicator()

        # System tray icon
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(self._make_tray_icon(running=False))
        tray_menu = QMenu()
        show_action = QAction("Show Panel", self)
        show_action.triggered.connect(self._restore_from_tray)
        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(quit_action)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.setToolTip("Strategy Control Panel")
        self._tray.show()
        tray_available = QSystemTrayIcon.isSystemTrayAvailable()
        logger.info(
            f"System tray — available={tray_available} "
            f"tray.isVisible()={self._tray.isVisible()}"
        )

        # Uptime refresh timer
        self._uptime_timer = QTimer(self)
        self._uptime_timer.setInterval(10_000)
        self._uptime_timer.timeout.connect(self._refresh_uptimes)
        self._uptime_timer.start()

    # --- Control tab ---

    def _build_control_tab(self, initial_config: Optional[str] = None) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)

        try:
            from market_monitor.utils.config_helpers import find_all_configs
            self._all_configs = list(find_all_configs())
        except Exception as e:
            logger.debug(f"Could not load config list: {e}")

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Config:"))
        self._config_combo = QComboBox()
        self._config_combo.setMinimumWidth(280)
        self._config_combo.setEditable(True)
        self._config_combo.addItems(self._all_configs)
        toolbar.addWidget(self._config_combo)
        add_btn = QPushButton("+ Add")
        add_btn.setStyleSheet(_BTN_SUCCESS)
        add_btn.clicked.connect(self._on_add_strategy)
        remove_btn = QPushButton("Remove")
        remove_btn.setStyleSheet(_BTN_DANGER)
        remove_btn.clicked.connect(self._on_remove_strategy)
        toolbar.addWidget(add_btn)
        toolbar.addStretch()
        toolbar.addWidget(remove_btn)
        layout.addLayout(toolbar)

        # Auto-restart toolbar
        ar_row = QHBoxLayout()
        ar_row.addWidget(QLabel("Selected strategy:"))
        self._auto_restart_check = QCheckBox("Auto-restart on crash")
        self._auto_restart_check.setToolTip(
            "If checked, the selected strategy restarts automatically after 3 s when it exits with a non-zero code."
        )
        self._auto_restart_check.setStyleSheet("color:#9090b0;")
        self._auto_restart_check.stateChanged.connect(self._on_auto_restart_toggled)
        ar_row.addWidget(self._auto_restart_check)
        ar_row.addStretch()
        layout.addLayout(ar_row)

        # Col indices: 0=Config 1=Status 2=Uptime 3=Trades 4=HF ms 5=Start 6=Stop
        self._strategy_table = QTableWidget(0, 7)
        self._strategy_table.setHorizontalHeaderLabels(
            ["Config", "Status", "Uptime", "Trades", "HF ms", "Start", "Stop"]
        )
        self._strategy_table.setAlternatingRowColors(False)
        hdr = self._strategy_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col, w_px in [(1, 115), (2, 75), (3, 65), (4, 60), (5, 65), (6, 65)]:
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
            self._strategy_table.setColumnWidth(col, w_px)
        self._strategy_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._strategy_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._strategy_table.verticalHeader().setVisible(False)
        self._strategy_table.setShowGrid(False)
        self._strategy_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self._strategy_table)

        if initial_config:
            self._add_instance(initial_config)

        return w

    def _on_add_strategy(self) -> None:
        name = self._config_combo.currentText().strip()
        if name:
            self._add_instance(name)

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

        self._lifecycle_stats[config_name] = {
            "market_trades": 0,
            "own_trades": 0,
            "last_hf_ms": None,
        }

        row = self._strategy_table.rowCount()
        self._strategy_table.insertRow(row)
        self._strategy_table.setItem(row, 0, QTableWidgetItem(config_name))
        self._strategy_table.setItem(row, 1, self._make_status_item("STOPPED"))
        self._strategy_table.setItem(row, 2, QTableWidgetItem("—"))
        self._strategy_table.setItem(row, 3, QTableWidgetItem("—"))
        self._strategy_table.setItem(row, 4, QTableWidgetItem("—"))

        start_btn = QPushButton("▶ Start")
        start_btn.setStyleSheet(_BTN_SUCCESS)
        start_btn.clicked.connect(lambda _checked, i=inst: i.start())
        stop_btn = QPushButton("■ Stop")
        stop_btn.setStyleSheet(_BTN_DANGER)
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(lambda _checked, i=inst: i.stop(self._redis_client))
        self._strategy_table.setCellWidget(row, 5, start_btn)
        self._strategy_table.setCellWidget(row, 6, stop_btn)

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
        self._strategy_table.setItem(row, 1, self._make_status_item(status))
        # Row background
        bg = _ROW_BG.get(status)
        for col in range(self._strategy_table.columnCount()):
            item = self._strategy_table.item(row, col)
            if item:
                if bg:
                    item.setBackground(bg)
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
        # Start/Stop button states
        start_btn = self._strategy_table.cellWidget(row, 5)
        stop_btn  = self._strategy_table.cellWidget(row, 6)
        if start_btn:
            start_btn.setEnabled(status not in ("RUNNING", "RESTARTING"))
        if stop_btn:
            stop_btn.setEnabled(status == "RUNNING")
        # Update tray + status bar
        self._update_status_bar()
        self._update_tray_icon()
        # Toast notification
        msg_map = {
            "RUNNING":    f"Strategy {inst.config_name} started",
            "STOPPED":    f"Strategy {inst.config_name} stopped",
            "ERROR":      f"Strategy {inst.config_name}: ERROR",
            "RESTARTING": f"Strategy {inst.config_name}: restarting…",
        }
        if status in msg_map and hasattr(self, "_tray"):
            self._tray.showMessage("Strategy Control Panel", msg_map[status],
                                   QSystemTrayIcon.Information, 3000)

    @staticmethod
    def _make_status_item(status: str) -> QTableWidgetItem:
        icon  = {"RUNNING": "●", "STOPPED": "○", "ERROR": "⚠", "RESTARTING": "↻"}.get(status, "○")
        color = {"RUNNING": "#27ae60", "ERROR": "#e74c3c", "RESTARTING": "#e67e22"}.get(status, "#888888")
        item = QTableWidgetItem(f"{icon} {status}")
        item.setForeground(QColor(color))
        bg = _ROW_BG.get(status)
        if bg:
            item.setBackground(bg)
        return item

    def _on_table_selection_changed(self) -> None:
        rows = self._strategy_table.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self._instances):
            inst = self._instances[rows[0].row()]
            if hasattr(self, "_auto_restart_check"):
                self._auto_restart_check.blockSignals(True)
                self._auto_restart_check.setChecked(inst.auto_restart)
                self._auto_restart_check.blockSignals(False)

    def _on_auto_restart_toggled(self, state: int) -> None:
        rows = self._strategy_table.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self._instances):
            inst = self._instances[rows[0].row()]
            inst.auto_restart = bool(state)

    def _refresh_uptimes(self) -> None:
        now = time.time()
        for row, inst in enumerate(self._instances):
            if inst.started_at is not None:
                uptime = _fmt_uptime(now - inst.started_at)
            else:
                uptime = "—"
            item = self._strategy_table.item(row, 2)
            if item:
                item.setText(uptime)

    def _refresh_stats_row(self, inst: StrategyInstance) -> None:
        row = self._instances.index(inst) if inst in self._instances else -1
        if row < 0:
            return
        stats = self._lifecycle_stats.get(inst.config_name, {})
        total_trades = stats.get("market_trades", 0) + stats.get("own_trades", 0)
        trades_txt = f"{total_trades:,}" if total_trades else "—"
        hf_ms = stats.get("last_hf_ms")
        hf_txt = f"{hf_ms:.0f}" if hf_ms is not None else "—"
        for col, txt in [(3, trades_txt), (4, hf_txt)]:
            item = self._strategy_table.item(row, col)
            if item:
                item.setText(txt)

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
        target_row.addWidget(QLabel("Target:"))
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(260)
        self._target_combo.addItem("(all / default channel)")
        target_row.addWidget(self._target_combo)
        target_row.addStretch()
        layout.addLayout(target_row)

        # Send Command — collapsible section
        self._send_toggle_btn = QPushButton("▶  Send Command (manuale)")
        self._send_toggle_btn.setStyleSheet(
            _BTN_NEUTRAL + "QPushButton{text-align:left;padding-left:10px;}"
        )
        self._send_toggle_btn.clicked.connect(self._toggle_send_command)
        layout.addWidget(self._send_toggle_btn)

        self._send_cmd_body = QWidget()
        body_layout = QVBoxLayout(self._send_cmd_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Action:"))
        self._cmd_action_combo = QComboBox()
        self._cmd_action_combo.setEditable(True)
        self._cmd_action_combo.setMinimumWidth(260)
        self._cmd_action_combo.addItem("(free text)", userData=None)
        self._cmd_action_combo.currentIndexChanged.connect(self._on_command_selected)
        action_row.addWidget(self._cmd_action_combo)
        action_row.addStretch()
        body_layout.addLayout(action_row)

        body_layout.addWidget(QLabel("JSON Payload:"))
        self._cmd_payload_edit = QTextEdit()
        self._cmd_payload_edit.setPlaceholderText("{}")
        self._cmd_payload_edit.setMaximumHeight(90)
        self._cmd_payload_edit.setFont(QFont("Courier New", 9))
        body_layout.addWidget(self._cmd_payload_edit)

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
        body_layout.addLayout(btn_row)

        self._send_cmd_body.setVisible(False)  # collapsed by default
        layout.addWidget(self._send_cmd_body)

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
                cmd_type = cmd.get("type", "button")

                if cmd_type == "float_input":
                    lbl = QLabel(f"{label}:")
                    grp_layout.addWidget(lbl)
                    spinbox = QDoubleSpinBox()
                    spinbox.setRange(cmd.get("min", 0.0), cmd.get("max", 1.0))
                    spinbox.setSingleStep(cmd.get("step", 0.05))
                    spinbox.setDecimals(2)
                    spinbox.setValue(cmd.get("default", 1.0))
                    spinbox.setFixedWidth(80)
                    if desc:
                        spinbox.setToolTip(desc)
                    grp_layout.addWidget(spinbox)
                    btn = QPushButton("Set")
                    btn.setFixedHeight(30)
                    btn.setFixedWidth(50)
                    btn.setStyleSheet(_BTN_PRIMARY)
                    btn.clicked.connect(
                        lambda _c, a=action, sp=spinbox, ch=channel: self._send_quick_command(a, {"value": sp.value()}, ch)
                    )
                    grp_layout.addWidget(btn)
                elif cmd_type == "isin_debug":
                    grp_layout.addWidget(QLabel("ISIN:"))
                    isin_edit = QLineEdit()
                    isin_edit.setPlaceholderText("es. IE00B4L5Y983")
                    isin_edit.setFixedWidth(160)
                    isin_edit.setFont(QFont("Courier New", 9))
                    if desc:
                        isin_edit.setToolTip(desc)
                    grp_layout.addWidget(isin_edit)
                    model_combo = QComboBox()
                    model_combo.setFixedWidth(110)
                    for m in cmd.get("models", ["cluster", "intraday", "index_cluster"]):
                        model_combo.addItem(m)
                    grp_layout.addWidget(model_combo)
                    btn = QPushButton(label)
                    btn.setFixedHeight(30)
                    btn.setStyleSheet(_BTN_PRIMARY)
                    if desc:
                        btn.setToolTip(desc)
                    btn.clicked.connect(
                        lambda _c, a=action, ie=isin_edit, mc=model_combo, ch=channel: (
                            self._send_quick_command(a, {"isin": ie.text().strip(), "model": mc.currentText()}, ch)
                            if ie.text().strip() else None
                        )
                    )
                    grp_layout.addWidget(btn)
                else:
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

    def _toggle_send_command(self) -> None:
        visible = not self._send_cmd_body.isVisible()
        self._send_cmd_body.setVisible(visible)
        self._send_toggle_btn.setText(
            "▼  Send Command (manuale)" if visible else "▶  Send Command (manuale)"
        )

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

        search_edit = QLineEdit()
        search_edit.setPlaceholderText("Search logs…")
        search_edit.setFixedWidth(180)
        search_edit.setStyleSheet(
            "QLineEdit{background:#1e1e2e;color:#d0d0d0;border:1px solid #3a3a5c;"
            "border-radius:4px;padding:2px 6px;}"
        )
        self._log_searches[key] = search_edit

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.setStyleSheet(_BTN_NEUTRAL)

        export_btn = QPushButton("Export…")
        export_btn.setFixedWidth(70)
        export_btn.setStyleSheet(_BTN_NEUTRAL)

        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setFont(QFont("Courier New", 9))
        edit.document().setMaximumBlockCount(5000)

        combo.currentIndexChanged.connect(
            lambda _idx, k=key, c=combo, e=edit: self._on_log_filter_changed(k, c, e)
        )
        search_edit.textChanged.connect(
            lambda txt, k=key, e=edit: self._on_log_search_changed(k, e, txt)
        )
        clear_btn.clicked.connect(lambda: (edit.clear(), self._log_buffers.__setitem__(key, [])))
        export_btn.clicked.connect(lambda: self._export_log(key))

        ctrl.addWidget(combo)
        ctrl.addSpacing(8)
        ctrl.addWidget(search_edit)
        ctrl.addSpacing(8)
        ctrl.addWidget(clear_btn)
        ctrl.addSpacing(4)
        ctrl.addWidget(export_btn)
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

    def _appendhtml_line(self, edit: QTextEdit, html: str) -> None:
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not edit.toPlainText():
            cursor.insertHtml(html)
        else:
            cursor.insertHtml("<br>" + html)
        edit.setTextCursor(cursor)
        edit.ensureCursorVisible()

    def _on_log_filter_changed(self, key: str, combo: QComboBox, edit: QTextEdit) -> None:
        search = (self._log_searches.get(key, QLineEdit()).text() or "").lower()
        self._rebuild_log_view(key, combo, edit, search)

    def _on_log_search_changed(self, key: str, edit: QTextEdit, search_txt: str) -> None:
        combo = self._log_filters.get(key, QComboBox())
        self._rebuild_log_view(key, combo, edit, search_txt.lower())

    def _rebuild_log_view(self, key: str, combo: QComboBox, edit: QTextEdit, search: str) -> None:
        min_rank = _LEVEL_RANK.get(combo.currentText(), 0)
        edit.clear()
        buf = self._log_buffers.get(key, [])
        first = True
        cursor = edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        for level, line in buf:
            if _LEVEL_RANK.get(level, 0) < min_rank:
                continue
            if search and search not in line.lower():
                continue
            html = self._format_log_html(level, line)
            if first:
                cursor.insertHtml(html)
                first = False
            else:
                cursor.insertHtml("<br>" + html)
        edit.setTextCursor(cursor)
        edit.ensureCursorVisible()

    def _export_log(self, key: str) -> None:
        combo = self._log_filters.get(key, QComboBox())
        search = (self._log_searches.get(key, QLineEdit()).text() or "").lower()
        min_rank = _LEVEL_RANK.get(combo.currentText(), 0)
        lines = [
            line for level, line in self._log_buffers.get(key, [])
            if _LEVEL_RANK.get(level, 0) >= min_rank
            and (not search or search in line.lower())
        ]
        if not lines:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", f"{key}_log.txt", "Text files (*.txt)"
        )
        if path:
            try:
                Path(path).write_text("\n".join(lines), encoding="utf-8")
            except Exception as e:
                logger.warning(f"Log export failed: {e}")

    def _on_instance_output(self, inst: StrategyInstance, text: str) -> None:
        for line in text.splitlines():
            if not line.strip():
                continue
            level = self._detect_level(line)
            prefixed = f"[{inst.config_name}] {line}"

            # Append to buffers
            self._log_buffers.setdefault(inst.config_name, []).append((level, f"[{inst.config_name}] {line}"))
            self._log_buffers["_all"].append((level, prefixed))

            # Parse lifecycle / stats markers
            self._parse_lifecycle_line(inst, line)

            # Check filters / search and append to visible edits
            inst_combo  = self._log_filters.get(inst.config_name, QComboBox())
            all_combo   = self._log_filters.get("_all", QComboBox())
            inst_search = (self._log_searches.get(inst.config_name, QLineEdit()).text() or "").lower()
            all_search  = (self._log_searches.get("_all", QLineEdit()).text() or "").lower()
            min_rank_inst = _LEVEL_RANK.get(inst_combo.currentText(), 0)
            min_rank_all  = _LEVEL_RANK.get(all_combo.currentText(), 0)
            line_lc = line.lower()
            if (_LEVEL_RANK.get(level, 0) >= min_rank_inst
                    and inst.config_name in self._log_tabs
                    and (not inst_search or inst_search in line_lc)):
                self._appendhtml_line(
                    self._log_tabs[inst.config_name],
                    self._format_log_html(level, f"[{inst.config_name}] {line}"),
                )
            if (_LEVEL_RANK.get(level, 0) >= min_rank_all
                    and (not all_search or all_search in prefixed.lower())):
                self._appendhtml_line(
                    self._log_all,
                    self._format_log_html(level, prefixed),
                )

    def _parse_lifecycle_line(self, inst: StrategyInstance, line: str) -> None:
        """Update live stats and trigger UI refresh when lifecycle/stats markers are found."""
        stats = self._lifecycle_stats.setdefault(inst.config_name, {
            "market_trades": 0, "own_trades": 0, "last_hf_ms": None,
        })

        lc_m = _LC_RE.search(line)
        if lc_m:
            event = lc_m.group(1)
            count = int(lc_m.group(2)) if lc_m.group(2) else 0
            if event == "started":
                inst.started_at = time.time()
            elif event == "market_trade":
                stats["market_trades"] = stats.get("market_trades", 0) + count
            elif event == "own_trade":
                stats["own_trades"] = stats.get("own_trades", 0) + count
            self._refresh_stats_row(inst)
            return

        st_m = _STATS_RE.search(line)
        if st_m:
            metric, val = st_m.group(1), float(st_m.group(2))
            if metric == "hf_update":
                stats["last_hf_ms"] = val * 1000
            self._refresh_stats_row(inst)

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
    # Slots — Redis
    # ------------------------------------------------------------------

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

    # ==================================================================
    # Config tab — YAML editor
    # ==================================================================

    def _build_config_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── top toolbar ──────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Strategy config:"))

        self._cfg_combo = QComboBox()
        self._cfg_combo.setMinimumWidth(280)
        self._cfg_combo.setEditable(True)
        self._cfg_combo.addItems(self._all_configs)
        self._cfg_combo.currentTextChanged.connect(self._on_cfg_combo_changed)
        toolbar.addWidget(self._cfg_combo)

        reload_btn = QPushButton("↺ Reload")
        reload_btn.setFixedWidth(80)
        reload_btn.setStyleSheet(_BTN_NEUTRAL)
        reload_btn.setToolTip("Discard unsaved changes and reload from disk")
        reload_btn.clicked.connect(self._reload_cfg_file)
        toolbar.addWidget(reload_btn)

        open_btn = QPushButton("📂 Open folder")
        open_btn.setFixedWidth(110)
        open_btn.setStyleSheet(_BTN_NEUTRAL)
        open_btn.setToolTip("Open the config folder in the file manager")
        open_btn.clicked.connect(self._open_cfg_folder)
        toolbar.addWidget(open_btn)

        toolbar.addStretch()
        root.addLayout(toolbar)

        # ── splitter: section list | editor ──────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left — section navigation
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)
        nav_lbl = QLabel("Sections")
        nav_lbl.setStyleSheet("color:#9090b0; font-size:11px; font-weight:bold;")
        left_layout.addWidget(nav_lbl)
        self._cfg_section_list = QListWidget()
        self._cfg_section_list.setStyleSheet(
            "QListWidget{background:#1e1e2e; border:1px solid #3a3a5c; border-radius:4px;}"
            "QListWidget::item{padding:4px 8px; color:#c0c0d8;}"
            "QListWidget::item:selected{background:#3a3a6e; color:#ffffff;}"
            "QListWidget::item:hover:!selected{background:#2a2a4e;}"
        )
        self._cfg_section_list.itemClicked.connect(self._on_cfg_section_clicked)
        left_layout.addWidget(self._cfg_section_list)
        left.setMinimumWidth(160)
        left.setMaximumWidth(240)
        splitter.addWidget(left)

        # Right — YAML text editor
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(0)

        self._yaml_editor = QTextEdit()
        self._yaml_editor.setFont(QFont("Courier New", 10))
        self._yaml_editor.setTabStopDistance(20)  # 2-space visual tab
        self._yaml_editor.setStyleSheet(
            "QTextEdit{background:#12121e; color:#d0d0d0;"
            "border:1px solid #3a3a5c; border-radius:4px;}"
        )
        self._yaml_highlighter = _YamlHighlighter(self._yaml_editor.document())
        self._yaml_editor.textChanged.connect(self._on_yaml_text_changed)
        right_layout.addWidget(self._yaml_editor)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # ── bottom bar: save / validate / status ─────────────────────
        bottom = QHBoxLayout()
        self._cfg_save_btn = QPushButton("💾 Save")
        self._cfg_save_btn.setStyleSheet(_BTN_SUCCESS)
        self._cfg_save_btn.setFixedWidth(90)
        self._cfg_save_btn.setEnabled(False)
        self._cfg_save_btn.clicked.connect(self._save_cfg_file)
        bottom.addWidget(self._cfg_save_btn)

        validate_btn = QPushButton("✔ Validate")
        validate_btn.setStyleSheet(_BTN_NEUTRAL)
        validate_btn.setFixedWidth(90)
        validate_btn.setToolTip("Check YAML syntax without saving")
        validate_btn.clicked.connect(self._validate_cfg_yaml)
        bottom.addWidget(validate_btn)

        self._cfg_status_lbl = QLabel("")
        self._cfg_status_lbl.setStyleSheet("font-size:11px; padding-left:8px;")
        bottom.addWidget(self._cfg_status_lbl)
        bottom.addStretch()

        self._cfg_path_lbl = QLabel("")
        self._cfg_path_lbl.setStyleSheet("color:#555577; font-size:10px;")
        bottom.addWidget(self._cfg_path_lbl)
        root.addLayout(bottom)

        # internal state
        self._cfg_path: Optional[Path] = None
        self._cfg_dirty: bool = False
        self._cfg_section_lines: Dict[str, int] = {}
        # debounce timer for section-list refresh
        self._cfg_update_timer = QTimer(self)
        self._cfg_update_timer.setSingleShot(True)
        self._cfg_update_timer.setInterval(300)
        self._cfg_update_timer.timeout.connect(self._refresh_cfg_sections)

        # Pre-load first config if available
        if self._all_configs:
            self._cfg_combo.setCurrentText(self._all_configs[0])
            self._load_cfg_file(self._all_configs[0])

        return w

    # ── config tab helpers ─────────────────────────────────────────────

    def _on_cfg_combo_changed(self, name: str) -> None:
        if self._cfg_dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                f"Discard unsaved changes to {self._cfg_path.name if self._cfg_path else 'current file'}?",
                QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ans != QMessageBox.Discard:
                # Restore the combo to the current file name
                if self._cfg_path:
                    self._cfg_combo.blockSignals(True)
                    self._cfg_combo.setCurrentText(self._cfg_path.stem)
                    self._cfg_combo.blockSignals(False)
                return
        self._load_cfg_file(name)

    def _load_cfg_file(self, config_name: str) -> None:
        """Resolve path, read text, populate editor and section list."""
        if not config_name.strip():
            return
        try:
            from market_monitor.utils.config_helpers import find_config
            path = find_config(config_name)
        except Exception as e:
            self._cfg_set_status(f"Config not found: {e}", error=True)
            return

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            self._cfg_set_status(f"Cannot read file: {e}", error=True)
            return

        self._cfg_path = path
        self._cfg_dirty = False
        self._yaml_editor.blockSignals(True)
        self._yaml_editor.setPlainText(text)
        self._yaml_editor.blockSignals(False)
        self._cfg_save_btn.setEnabled(False)
        self._cfg_path_lbl.setText(str(path))
        self._cfg_set_status("Loaded.", error=False)
        self._refresh_cfg_sections()

    def _reload_cfg_file(self) -> None:
        name = self._cfg_combo.currentText().strip()
        if name:
            self._cfg_dirty = False  # bypass unsaved-changes check
            self._load_cfg_file(name)

    def _refresh_cfg_sections(self) -> None:
        """Parse top-level YAML keys and populate the section list."""
        self._cfg_section_lines.clear()
        self._cfg_section_list.clear()
        text = self._yaml_editor.toPlainText()
        for lineno, line in enumerate(text.splitlines()):
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
            if m:
                key = m.group(1)
                self._cfg_section_lines[key] = lineno
                item = QListWidgetItem(key)
                self._cfg_section_list.addItem(item)

    def _on_cfg_section_clicked(self, item: QListWidgetItem) -> None:
        """Scroll editor to the clicked section."""
        key = item.text()
        lineno = self._cfg_section_lines.get(key, 0)
        doc = self._yaml_editor.document()
        block = doc.findBlockByLineNumber(lineno)
        cursor = QTextCursor(block)
        self._yaml_editor.setTextCursor(cursor)
        self._yaml_editor.ensureCursorVisible()
        self._yaml_editor.setFocus()

    def _on_yaml_text_changed(self) -> None:
        if not self._cfg_dirty:
            self._cfg_dirty = True
            self._cfg_save_btn.setEnabled(True)
            self._cfg_set_status("Unsaved changes", error=False, warning=True)
        self._cfg_update_timer.start()  # debounced section refresh

    def _validate_cfg_yaml(self) -> bool:
        """Validate YAML syntax. Returns True if valid."""
        try:
            import yaml
            yaml.safe_load(self._yaml_editor.toPlainText())
            self._cfg_set_status("✔ Valid YAML", error=False)
            return True
        except Exception as e:
            # Try to pinpoint the line number
            msg = str(e)
            self._cfg_set_status(f"⚠ YAML error: {msg}", error=True)
            # Try to jump to the error line
            m = re.search(r"line (\d+)", msg)
            if m:
                lineno = int(m.group(1)) - 1
                block = self._yaml_editor.document().findBlockByLineNumber(lineno)
                cursor = QTextCursor(block)
                self._yaml_editor.setTextCursor(cursor)
                self._yaml_editor.ensureCursorVisible()
            return False

    def _save_cfg_file(self) -> None:
        if self._cfg_path is None:
            return
        if not self._validate_cfg_yaml():
            ans = QMessageBox.question(
                self, "Invalid YAML",
                "The file contains YAML errors. Save anyway?",
                QMessageBox.Save | QMessageBox.Cancel,
            )
            if ans != QMessageBox.Save:
                return
        try:
            self._cfg_path.write_text(
                self._yaml_editor.toPlainText(), encoding="utf-8"
            )
            self._cfg_dirty = False
            self._cfg_save_btn.setEnabled(False)
            self._cfg_set_status(f"Saved → {self._cfg_path.name}", error=False)
        except Exception as e:
            self._cfg_set_status(f"Save error: {e}", error=True)

    def _open_cfg_folder(self) -> None:
        folder = self._cfg_path.parent if self._cfg_path else None
        if folder and folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _cfg_set_status(self, msg: str, *, error: bool, warning: bool = False) -> None:
        if error:
            style = "color:#e74c3c; font-size:11px; padding-left:8px;"
        elif warning:
            style = "color:#e67e22; font-size:11px; padding-left:8px;"
        else:
            style = "color:#27ae60; font-size:11px; padding-left:8px;"
        self._cfg_status_lbl.setStyleSheet(style)
        self._cfg_status_lbl.setText(msg)
