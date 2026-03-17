import logging

import pandas as pd
from PyQt5.QtCore import pyqtSignal, Qt, QSettings
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QLabel,
                             QInputDialog)

from market_monitor.gui.implementations.PyQt5Dashboard.widgets.chart_widget import ChartWidget
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.flow_monitor_widget import FlowMonitorWidget
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.pivot_table import PivotTableWidget
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.groupby_widget import GroupByWidget

logger = logging.getLogger(__name__)

_SETTINGS_ORG = "MarketMonitor"
_SETTINGS_APP = "DetachedWindows"

def _make_renameable_label(label: QLabel, window: QMainWindow, prefix: str = "") -> None:
    """Rende un info_label rinominabile con doppio clic."""
    label.setCursor(Qt.PointingHandCursor)
    label.setToolTip("Double-click to rename")

    def on_double_click(event):
        current = window.windowTitle()
        new_name, ok = QInputDialog.getText(
            window, "Rename", "New name:", text=current
        )
        if ok and new_name.strip():
            window.setWindowTitle(new_name.strip())
            label.setText(new_name.strip())

    label.mouseDoubleClickEvent = on_double_click


class DetachedPivotWindow(QMainWindow):
    """Finestra separata per pivot table con aggiornamento dati"""

class BaseDetachedWindow(QMainWindow):
    """
    Base class per tutte le finestre detached della dashboard.

    Fornisce:
    - Rinomina del titolo: doppio-click sul label info oppure
      Ctrl+R → dialog di input → aggiorna windowTitle + info_label
    - Geometria auto-persistente: salva posizione/dimensione in QSettings
      ogni volta che la finestra viene ridimensionata o spostata;
      al riavvio la posizione viene ripristinata automaticamente.
    """

    # Subclass must set these before calling _finish_init()
    _window_type: str = "base"        # e.g. "pivot", "chart", "flow", "groupby"
    _default_title: str = "Window"

    def __init__(self, window_number: int, parent=None):
        super().__init__(parent)
        self.window_number = window_number
        self._custom_title: str = ""   # vuoto = usa il titolo di default
        self._settings_key = f"{self._window_type}_{window_number}"

        # Override di show() posticipato: viene chiamato dopo _finish_init()

    # ------------------------------------------------------------------
    # Inizializzazione (da chiamare alla fine di ogni sottoclasse)
    # ------------------------------------------------------------------

    def _finish_init(self):
        """
        Da chiamare alla fine di __init__ delle sottoclassi, dopo aver
        creato info_label.  Ripristina titolo e geometria da QSettings.
        """
        self._setup_rename_shortcut()
        self._restore_from_settings()

    # ------------------------------------------------------------------
    # Rinomina
    # ------------------------------------------------------------------

    def _setup_rename_shortcut(self):
        """Collega il doppio-click sull'info_label al dialog di rinomina."""
        if hasattr(self, 'info_label'):
            self.info_label.setToolTip("Doppio-click per rinominare la finestra")
            self.info_label.mouseDoubleClickEvent = lambda _e: self.rename_window()

    def rename_window(self):
        """Apre un dialog per cambiare il titolo della finestra."""
        current = self._custom_title or self._default_title
        new_title, ok = QInputDialog.getText(
            self, "Rinomina finestra", "Nuovo titolo:", text=current
        )
        if ok and new_title.strip():
            self.set_custom_title(new_title.strip())

    def set_custom_title(self, title: str):
        """Imposta un titolo personalizzato (stringa vuota = default)."""
        self._custom_title = title
        self.setWindowTitle(title if title else self._default_title)
        if hasattr(self, 'info_label'):
            # Mantiene il suffisso informativo già presente nel testo label
            current_text = self.info_label.text()
            # Sostituisce solo la parte iniziale (fino al primo '|' o tutto)
            if '|' in current_text:
                suffix = current_text[current_text.index('|'):]
                self.info_label.setText(f"{self.setWindowTitle.__self__.windowTitle()}{suffix}")
            # Aggiorna QSettings
        self._save_to_settings()
        logger.debug(f"[{self._settings_key}] Title set to: {self.windowTitle()!r}")

    # ------------------------------------------------------------------
    # Persistenza geometria via QSettings
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def _save_to_settings(self):
        """Salva geometria e titolo personalizzato in QSettings."""
        s = self._settings()
        s.beginGroup(self._settings_key)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("custom_title", self._custom_title)
        s.endGroup()
        s.sync()

    def _restore_from_settings(self):
        """Ripristina geometria e titolo da QSettings se presenti."""
        s = self._settings()
        s.beginGroup(self._settings_key)
        geometry = s.value("geometry")
        custom_title = s.value("custom_title", "")
        s.endGroup()

        if geometry:
            self.restoreGeometry(geometry)
            logger.debug(f"[{self._settings_key}] Geometry restored from QSettings")

        if custom_title:
            self._custom_title = custom_title
            self.setWindowTitle(custom_title)
            logger.debug(f"[{self._settings_key}] Title restored: {custom_title!r}")

    # ------------------------------------------------------------------
    # Intercettazione eventi Qt per auto-salvataggio geometria
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._save_to_settings()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._save_to_settings()

    def closeEvent(self, event):
        self._save_to_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helper per le sottoclassi: aggiorna testo info_label
    # ------------------------------------------------------------------

    def _set_info_text(self, suffix: str):
        """Aggiorna info_label con titolo corrente + suffisso informativo."""
        title = self.windowTitle()
        text = f"{title}  |  {suffix}" if suffix else title
        if hasattr(self, 'info_label'):
            self.info_label.setText(text)


# ===========================================================================
# Finestre concrete
# ===========================================================================

class DetachedPivotWindow(BaseDetachedWindow):
    """Finestra separata per pivot table con aggiornamento dati."""

    data_update_needed = pyqtSignal(object)

    _window_type = "pivot"
    _default_title = "Pivot Table #{n}"

    def __init__(self, source_data, window_number, parent=None):
        super().__init__(window_number, parent)
        self._default_title = f"Pivot Table #{window_number}"
        self.setWindowTitle(self._default_title)
        self.setGeometry(200 + (window_number * 30), 200 + (window_number * 30), 900, 600)
        self.setMinimumSize(300, 200)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel(f"Pivot Table #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e0f7fa;")
        _make_renameable_label(self.info_label, self)
        self.info_label = QLabel(self._default_title)
        self.info_label.setStyleSheet(
            "font-weight: bold; padding: 5px; background-color: #e0f7fa; cursor: pointer;"
        )
        layout.addWidget(self.info_label)

        self.pivot_widget = PivotTableWidget()
        self.pivot_widget.set_source_data(source_data)
        layout.addWidget(self.pivot_widget)

        self._finish_init()

    def update_source_data(self, data: pd.DataFrame):
        self.pivot_widget.set_source_data(data)
        self._set_info_text(f"Source rows: {len(data)}")

    def clear_data(self):
        self.pivot_widget.clear_pivot()
        self.pivot_widget.set_source_data(pd.DataFrame())


class DetachedChartWindow(BaseDetachedWindow):
    """Finestra separata per grafici."""

    _window_type = "chart"
    _default_title = "Charts #{n}"

    def __init__(self, source_data, window_number, parent=None):
        super().__init__(window_number, parent)
        self._default_title = f"Charts #{window_number}"
        self.setWindowTitle(self._default_title)
        self.setGeometry(300 + (window_number * 30), 300 + (window_number * 30), 1000, 700)
        self.setMinimumSize(300, 200)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel(f"Charts #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #fff3e0;")
        _make_renameable_label(self.info_label, self)
        self.info_label = QLabel(self._default_title)
        self.info_label.setStyleSheet(
            "font-weight: bold; padding: 5px; background-color: #fff3e0; cursor: pointer;"
        )
        layout.addWidget(self.info_label)

        self.chart_widget = ChartWidget()
        self.chart_widget.set_data(source_data)
        layout.addWidget(self.chart_widget)

        self._finish_init()

    def update_source_data(self, data: pd.DataFrame):
        self.chart_widget.set_data(data)
        self._set_info_text(f"Source rows: {len(data)}")


class DetachedFlowWindow(BaseDetachedWindow):
    """Finestra separata per Flow Monitor."""

    _window_type = "flow"
    _default_title = "Flow Monitor #{n}"

    def __init__(self, window_number, parent=None):
        super().__init__(window_number, parent)
        self._default_title = f"Flow Monitor #{window_number}"
        self.setWindowTitle(self._default_title)
        self.setGeometry(400 + (window_number * 30), 400 + (window_number * 30), 500, 700)
        self.setMinimumSize(300, 200)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel(f"Flow Monitor #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e3f2fd;")
        _make_renameable_label(self.info_label, self)
        self.info_label = QLabel(self._default_title)
        self.info_label.setStyleSheet(
            "font-weight: bold; padding: 5px; background-color: #e3f2fd; cursor: pointer;"
        )
        layout.addWidget(self.info_label)

        self.flow_widget = FlowMonitorWidget(auto_hide=False)
        layout.addWidget(self.flow_widget)

        self._finish_init()

    def add_flow(self, flow_data: dict):
        self.flow_widget.update_flows([flow_data])
        flow_count = len(self.flow_widget.flow_cards)
        self._set_info_text(f"Flows: {flow_count}")

    def clear_all(self):
        self.flow_widget.clear_all()
        self._set_info_text("Flows: 0")


class DetachedGroupByWindow(BaseDetachedWindow):
    """Finestra separata per GroupBy multi-colonna con aggregazioni."""

    _window_type = "groupby"
    _default_title = "GroupBy #{n}"

    def __init__(self, source_data, window_number, parent=None):
        super().__init__(window_number, parent)
        self._default_title = f"GroupBy #{window_number}"
        self.setWindowTitle(self._default_title)
        self.setGeometry(250 + (window_number * 30), 250 + (window_number * 30), 900, 600)
        self.setMinimumSize(300, 200)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.info_label = QLabel(f"GroupBy #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e8f5e9;")
        _make_renameable_label(self.info_label, self)
        self.info_label = QLabel(self._default_title)
        self.info_label.setStyleSheet(
            "font-weight: bold; padding: 5px; background-color: #e8f5e9; cursor: pointer;"
        )
        layout.addWidget(self.info_label)

        self.groupby_widget = GroupByWidget()
        if isinstance(source_data, pd.DataFrame) and not source_data.empty:
            self.groupby_widget.set_source_data(source_data)
        layout.addWidget(self.groupby_widget)

        self._finish_init()

    def update_source_data(self, data: pd.DataFrame):
        self.groupby_widget.set_source_data(data)
        self._set_info_text(f"Source rows: {len(data)}")

    def clear_data(self):
        self.groupby_widget.clear_groupby()
        self.groupby_widget.source_data = pd.DataFrame()
