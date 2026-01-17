"""
Widget TradeTable con filtri avanzati AND/OR e infinite scrolling
"""

import datetime
from typing import Optional, List

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QGroupBox, QHeaderView, QSpinBox, QCheckBox,
    QMenu, QAction, QWidgetAction
)

from market_monitor.gui.implementations.PyQt5Dashboard.common import safe_concat
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup


class TradeTableWidget(QWidget):
    """
    Visualizza trades con:
    - filtri avanzati AND/OR
    - infinite scrolling
    - formatting per-colonna
    """

    filtered_data_changed = pyqtSignal(pd.DataFrame)

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        max_rows: int = 1000,
        dedup_column: str = "trade_index",
        datetime_columns="timestamp",
        datetime_format: str = "%H:%M:%S.%f",
        parent=None,
    ):
        super().__init__(parent)

        # ---- Data ----
        self.all_data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()

        self.visible_columns = columns or []
        self.max_rows = max_rows
        self.dedup_column = dedup_column

        self.datetime_columns = (
            datetime_columns if isinstance(datetime_columns, list)
            else [datetime_columns]
        )
        self.datetime_format = datetime_format
        self.column_decimals: dict[str, int] = {}

        # ---- Filters ----
        self.active_filter: Optional[FilterGroup] = None

        # ---- Infinite scroll ----
        self.displayed_rows = 0
        self.rows_per_batch = 100
        self.is_loading = False

        self._setup_ui()

    # ==========================================================
    # UI
    # ==========================================================
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ---------- CONTROLS ----------
        controls = QGroupBox("Table Controls")
        controls_layout = QHBoxLayout()

        self.autoscroll_checkbox = QCheckBox("Auto-scroll")
        self.autoscroll_checkbox.setChecked(True)
        controls_layout.addWidget(self.autoscroll_checkbox)

        self.advanced_filter_btn = QPushButton("🔍 Advanced Filter")
        self.advanced_filter_btn.clicked.connect(
            self._show_advanced_filter_dialog
        )
        controls_layout.addWidget(self.advanced_filter_btn)

        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_all_filters)
        controls_layout.addWidget(clear_btn)

        controls_layout.addStretch()

        self.filter_info_label = QLabel("No filters active")
        controls_layout.addWidget(self.filter_info_label)

        self.info_label = QLabel("No data")
        controls_layout.addWidget(self.info_label)

        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # ---------- TABLE ----------
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(
            self._show_header_context_menu
        )

        self.table.verticalScrollBar().valueChanged.connect(
            self._on_scroll
        )

        layout.addWidget(self.table)

    # ==========================================================
    # HEADER CONTEXT MENU
    # ==========================================================
    def _show_header_context_menu(self, pos: QPoint):
        col = self.table.horizontalHeader().logicalIndexAt(pos)
        if col < 0:
            return

        name = self.table.horizontalHeaderItem(col).text()
        if name not in self.filtered_data.columns:
            return

        series = self.filtered_data[name]
        menu = QMenu(self)

        title = QAction(f"📊 {name}", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()

        if pd.api.types.is_numeric_dtype(series):
            spin = QSpinBox()
            spin.setRange(0, 6)
            spin.setValue(self.column_decimals.get(name, 2))

            spin.valueChanged.connect(
                lambda v, c=name: self._set_column_decimals(c, v)
            )

            w = QWidget()
            l = QHBoxLayout(w)
            l.addWidget(QLabel("Decimals"))
            l.addWidget(spin)

            act = QWidgetAction(self)
            act.setDefaultWidget(w)
            menu.addAction(act)

        menu.exec_(QCursor.pos())

    # ==========================================================
    # FILTERS
    # ==========================================================
    def _show_advanced_filter_dialog(self):
        if self.all_data.empty:
            return

        dlg = AdvancedFilterDialog(
            columns=list(self.all_data.columns),
            data=self.all_data,
            current_filter=self.active_filter,
            parent=self,
        )

        if dlg.exec_():
            self.active_filter = dlg.get_filter()
            self._apply_filters()

    def _clear_all_filters(self):
        self.active_filter = None
        self._apply_filters()

    def _apply_filters(self):
        if self.all_data.empty:
            self.filtered_data = pd.DataFrame()
            self._refresh_view()
            return

        if not self.active_filter or not self.active_filter.conditions:
            self.filtered_data = self.all_data.copy()
        else:
            try:
                mask = self.active_filter.apply(self.all_data)
                
                # Assicuriamoci che mask sia una Series con gli stessi indici
                if isinstance(mask, pd.Series):
                    # Reindex per garantire corrispondenza con all_data
                    mask = mask.reindex(self.all_data.index, fill_value=False)
                
                self.filtered_data = self.all_data[mask]
                
            except Exception as e:
                # Mostra errore all'utente invece di crashare
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Filter Error",
                    f"Failed to apply filter: {str(e)}\n\n"
                    f"Please check your filter conditions.\n"
                    f"Common issues:\n"
                    f"- Invalid data type (e.g., text in numeric/date fields)\n"
                    f"- Invalid date format\n"
                    f"- Malformed comparison values"
                )
                # Reset al filtro precedente (nessun filtro)
                self.active_filter = None
                self.filtered_data = self.all_data.copy()

        self.displayed_rows = 0
        self._refresh_view()
        self.filtered_data_changed.emit(self.filtered_data)

    # ==========================================================
    # DATA UPDATE
    # ==========================================================
    def update_data(self, df: pd.DataFrame):
        if df is None or df.empty:
            return

        if self.all_data.empty:
            self.all_data = df.copy()
        else:
            self.all_data = safe_concat(
                [self.all_data, df], ignore_index=True
            )

            if self.dedup_column in self.all_data.columns:
                self.all_data.drop_duplicates(
                    subset=[self.dedup_column],
                    keep="last",
                    inplace=True,
                )

        if not self.visible_columns:
            self.visible_columns = list(self.all_data.columns)

        self._apply_filters()

    # ==========================================================
    # TABLE RENDERING
    # ==========================================================
    def _refresh_view(self):
        """Reset completo - solo quando filtri cambiano"""
        self.table.setRowCount(0)  # ✅ Pulisci TUTTO
        self.table.setColumnCount(0)
        self.displayed_rows = 0
        self._load_more_rows()

    def _load_more_rows(self):
        if self.is_loading or self.filtered_data.empty:
            return

        df = self.filtered_data[self.visible_columns]
        if self.displayed_rows >= len(df):
            return

        self.is_loading = True
        start = self.displayed_rows
        end = min(start + self.rows_per_batch, len(df))
        self.displayed_rows = end

        self._render_rows(df, start, end)
        self._update_info()
        self.is_loading = False

    def _render_rows(self, df: pd.DataFrame, start: int, end: int):
        self.table.setSortingEnabled(False)

        self.table.setRowCount(end)
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())

        side_idx = df.columns.get_loc("side") if "side" in df else None
        own_idx = df.columns.get_loc("own_trade") if "own_trade" in df else None

        for i, row in enumerate(df.iloc[start:end].itertuples(index=False), start):
            row_color = None
            is_own = False

            if side_idx is not None:
                if str(row[side_idx]).upper() == "BID":
                    row_color = QColor(220, 235, 255)
                elif str(row[side_idx]).upper() == "ASK":
                    row_color = QColor(255, 235, 235)

            if own_idx is not None:
                is_own = bool(row[own_idx])

            for j, val in enumerate(row):
                text = self._format_value(df.columns[j], val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)

                if row_color:
                    item.setBackground(row_color)
                if is_own:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)

                self.table.setItem(i, j, item)

        self.table.setSortingEnabled(True)

    # ==========================================================
    # FORMATTING
    # ==========================================================
    def _format_value(self, col: str, value) -> str:
        if pd.isna(value):  
            return ""

        # Priorità: Se è un datetime/Timestamp, formattalo sempre come tale
        if isinstance(value, (pd.Timestamp, datetime.datetime)):
            return self._format_datetime(value)
        
        # Se la colonna è marcata come datetime, formattala
        if col in self.datetime_columns:
            return self._format_datetime(value)

        # Numeri float
        if isinstance(value, float):
            d = self.column_decimals.get(col, 2)
            return f"{value:.{d}f}".rstrip("0").rstrip(".")

        return str(value)

    def _format_datetime(self, value) -> str:
        """Formatta datetime come HH:MM:SS.fff"""
        try:
            # Se è già un pandas Timestamp o datetime, usalo direttamente
            if isinstance(value, (pd.Timestamp, datetime.datetime)):
                dt = value
            # Se è stringa, converti
            elif isinstance(value, str):
                dt = pd.to_datetime(value)
            # Se è numero (Unix timestamp), converti
            elif isinstance(value, (int, float)):
                # Gestisce sia secondi che millisecondi
                dt = pd.to_datetime(value, unit='s' if value < 1e12 else 'ms')
            else:
                return str(value)
            
            # Formatta con il formato richiesto
            txt = dt.strftime(self.datetime_format)
            # Taglia microsecondi se richiesto
            return txt[:-3] if "%f" in self.datetime_format else txt
            
        except Exception as e:
            return str(value)

    # ==========================================================
    # SCROLL
    # ==========================================================
    def _on_scroll(self, v):
        sb = self.table.verticalScrollBar()
        if v >= sb.maximum() * 0.9:
            self._load_more_rows()

    # ==========================================================
    # API
    # ==========================================================
    def set_visible_columns(self, cols: List[str]):
        """Imposta le colonne visibili e aggiorna la vista"""
        if not cols:
            return

        # Aggiorna le colonne visibili
        self.visible_columns = cols

        # Forza il refresh completo della vista
        self.displayed_rows = 0
        self._refresh_view()

        # Opzionale: emetti il segnale per notificare il cambio
        # (solo se filtri erano già applicati)
        if not self.filtered_data.empty:
            self.filtered_data_changed.emit(self.filtered_data)

    def get_available_columns(self) -> List[str]:
        return self.all_data.columns.tolist()

    def get_visible_columns(self) -> List[str]:
        return self.visible_columns

    def clear(self):
        self.all_data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()
        self.active_filter = None
        self.displayed_rows = 0
        self.table.clear()
        self.info_label.setText("No data")

    # ==========================================================
    # UI HELPERS
    # ==========================================================
    def _update_info(self):
        tot = len(self.all_data)
        flt = len(self.filtered_data)
        self.info_label.setText(
            f"Total: {tot} | Filtered: {flt} | Showing: {self.displayed_rows}"
        )

    def _set_column_decimals(self, col: str, d: int):
        self.column_decimals[col] = d
        self._refresh_view()
