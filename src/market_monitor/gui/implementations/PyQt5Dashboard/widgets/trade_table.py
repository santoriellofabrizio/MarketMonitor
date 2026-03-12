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
    QMenu, QAction, QWidgetAction, QScrollArea, QFrame
)

from market_monitor.gui.implementations.PyQt5Dashboard.common import safe_concat
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup

# Parole chiave che indicano colonne con valori "con segno" (positivo/negativo)
# → scala divergente rosso-bianco-verde
_CF_SIGNED_KEYWORDS = frozenset({
    "pnl", "profit", "return", "change", "gain", "loss",
    "delta", "diff", "net", "spread", "edge",
})


class NumericTableWidgetItem(QTableWidgetItem):
    """
    QTableWidgetItem che supporta sorting numerico.
    Memorizza il valore originale e sovrascrive __lt__ per confronto corretto.
    """

    def __init__(self, display_text: str, sort_value=None):
        super().__init__(display_text)
        self._sort_value = sort_value

    def __lt__(self, other):
        if not isinstance(other, NumericTableWidgetItem):
            return super().__lt__(other)

        # Se entrambi hanno valori numerici, confronta numericamente
        self_val = self._sort_value
        other_val = other._sort_value

        # Gestisci None/NaN come minori di tutto
        self_is_none = self_val is None or (isinstance(self_val, float) and pd.isna(self_val))
        other_is_none = other_val is None or (isinstance(other_val, float) and pd.isna(other_val))

        if self_is_none and other_is_none:
            return False
        if self_is_none:
            return True  # None va in fondo
        if other_is_none:
            return False

        # Confronto numerico se entrambi sono numeri
        if isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
            return self_val < other_val

        # Confronto datetime
        if isinstance(self_val, (pd.Timestamp, datetime.datetime)) and isinstance(other_val, (pd.Timestamp, datetime.datetime)):
            return self_val < other_val

        # Fallback a confronto stringa
        return self.text() < other.text()


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

        # ---- Conditional Formatting ----
        self.conditional_formatting: bool = False

        # ---- Filters ----
        self.active_filter: Optional[FilterGroup] = None
        # Filtri per valori colonna: {col_name: set di valori esclusi}
        self._column_value_filters: dict[str, set] = {}

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

        self.cf_btn = QPushButton("CF: Off")
        self.cf_btn.setCheckable(True)
        self.cf_btn.setToolTip(
            "Conditional Formatting: colora le celle numeriche con gradienti.\n"
            "• Colonne PnL/profit/return/change → scala divergente rosso-bianco-verde\n"
            "• Altre colonne numeriche → scala sequenziale bianco-blu"
        )
        self.cf_btn.clicked.connect(self._toggle_cf)
        controls_layout.addWidget(self.cf_btn)

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
        header.setSectionsMovable(True)
        header.sectionMoved.connect(self._on_section_moved)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(
            self._show_header_context_menu
        )
        # Doppio click per filtro valori
        header.sectionDoubleClicked.connect(self._show_column_filter_menu)

        self.table.verticalScrollBar().valueChanged.connect(
            self._on_scroll
        )

        layout.addWidget(self.table)

    # ==========================================================
    # COLUMN REORDER (header drag)
    # ==========================================================
    def _on_section_moved(self, logical: int, old_visual: int, new_visual: int):
        """Aggiorna visible_columns quando l'utente trascina un'intestazione di colonna."""
        header = self.table.horizontalHeader()
        n = self.table.columnCount()
        new_order = [
            self.table.horizontalHeaderItem(header.logicalIndex(vi)).text()
            for vi in range(n)
        ]
        self.visible_columns = new_order
        self._refresh_view()

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

    def _show_column_filter_menu(self, col_index: int):
        """Mostra menu con checkbox per filtrare i valori della colonna."""
        if col_index < 0 or self.all_data.empty:
            return

        header_item = self.table.horizontalHeaderItem(col_index)
        if not header_item:
            return

        col_name = header_item.text()
        if col_name not in self.all_data.columns:
            return

        # Ottieni valori unici (usa all_data per avere tutti i valori possibili)
        unique_values = self.all_data[col_name].dropna().unique()

        # Limita a 100 valori per performance
        if len(unique_values) > 100:
            unique_values = unique_values[:100]

        # Ordina i valori
        try:
            unique_values = sorted(unique_values)
        except TypeError:
            unique_values = sorted(unique_values, key=str)

        # Valori attualmente esclusi (copia per stato temporaneo)
        excluded = self._column_value_filters.get(col_name, set()).copy()
        # Stato temporaneo per questo menu
        temp_excluded = [excluded]  # Lista per permettere modifica nella closure

        # Crea menu
        menu = QMenu(self)
        menu.setMinimumWidth(200)

        # Titolo
        title = QAction(f"🔍 Filter: {col_name}", self)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()

        # Select All / Unselect All (usando QPushButton per non chiudere il menu)
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(5, 2, 5, 2)
        btn_layout.setSpacing(5)

        select_all_btn = QPushButton("✓ Select All")
        select_all_btn.clicked.connect(
            lambda: self._filter_select_all_temp(checkboxes, temp_excluded)
        )
        btn_layout.addWidget(select_all_btn)

        unselect_all_btn = QPushButton("✗ Unselect All")
        unselect_all_btn.clicked.connect(
            lambda: self._filter_unselect_all_temp(unique_values, checkboxes, temp_excluded)
        )
        btn_layout.addWidget(unselect_all_btn)

        btn_action = QWidgetAction(self)
        btn_action.setDefaultWidget(btn_container)
        menu.addAction(btn_action)

        menu.addSeparator()

        # Container scrollabile per checkbox
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(2)

        checkboxes = []
        for val in unique_values:
            display_text = str(val) if not pd.isna(val) else "(empty)"
            cb = QCheckBox(display_text)
            cb.setChecked(val not in temp_excluded[0])
            cb.stateChanged.connect(
                lambda state, v=val: self._on_filter_checkbox_temp(v, state, temp_excluded)
            )
            scroll_layout.addWidget(cb)
            checkboxes.append((val, cb))

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_action = QWidgetAction(self)
        scroll_action.setDefaultWidget(scroll_area)
        menu.addAction(scroll_action)

        menu.addSeparator()

        # Bottone Filter per applicare
        filter_btn = QPushButton("Filter")
        filter_btn.clicked.connect(
            lambda: self._apply_column_filter(col_name, temp_excluded[0], menu)
        )
        filter_action = QWidgetAction(self)
        filter_action.setDefaultWidget(filter_btn)
        menu.addAction(filter_action)

        # Mostra menu alla posizione dell'header
        header = self.table.horizontalHeader()
        pos = header.mapToGlobal(QPoint(header.sectionPosition(col_index), header.height()))
        menu.exec_(pos)

    def _on_filter_checkbox_temp(self, value, state: int, temp_excluded: list):
        """Aggiorna stato temporaneo (non applica ancora il filtro)."""
        if state == Qt.Checked:
            temp_excluded[0].discard(value)
        else:
            temp_excluded[0].add(value)

    def _filter_select_all_temp(self, checkboxes, temp_excluded: list):
        """Seleziona tutti (stato temporaneo)."""
        temp_excluded[0].clear()
        for val, cb in checkboxes:
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)

    def _filter_unselect_all_temp(self, values, checkboxes, temp_excluded: list):
        """Deseleziona tutti (stato temporaneo)."""
        temp_excluded[0] = set(values)
        for val, cb in checkboxes:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _apply_column_filter(self, col_name: str, excluded: set, menu: QMenu):
        """Applica il filtro colonna e chiude il menu."""
        if excluded:
            self._column_value_filters[col_name] = excluded.copy()
        elif col_name in self._column_value_filters:
            del self._column_value_filters[col_name]

        self._apply_filters()
        self._update_filter_info()
        menu.close()

    def _update_filter_info(self):
        """Aggiorna label info filtri."""
        filters_active = []

        if self.active_filter and self.active_filter.conditions:
            filters_active.append("Advanced")

        for col_name, excluded in self._column_value_filters.items():
            if excluded:
                filters_active.append(f"{col_name}({len(excluded)})")

        if filters_active:
            self.filter_info_label.setText(f"Filters: {', '.join(filters_active)}")
        else:
            self.filter_info_label.setText("No filters active")

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
        self._column_value_filters.clear()
        self._apply_filters()
        self._update_filter_info()

    def _apply_filters(self):
        if self.all_data.empty:
            self.filtered_data = pd.DataFrame()
            self._refresh_view()
            return

        # 1. Applica filtro avanzato (se presente)
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

        # 2. Applica filtri per valori colonna (checkbox)
        if self._column_value_filters:
            for col_name, excluded_values in self._column_value_filters.items():
                if col_name in self.filtered_data.columns and excluded_values:
                    self.filtered_data = self.filtered_data[
                        ~self.filtered_data[col_name].isin(excluded_values)
                    ]

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
                sort_cols = (
                    ['timestamp', self.dedup_column]
                    if 'timestamp' in self.all_data.columns
                    else [self.dedup_column]
                )
                try:
                    self.all_data = self.all_data.sort_values(
                        by=sort_cols, ascending=True, na_position='first'
                    )
                except Exception:
                    pass
                self.all_data = (
                    self.all_data
                    .groupby(self.dedup_column, sort=False)
                    .last()
                    .reset_index()
                )

                # Re-sort descending so newest trades appear at top
                try:
                    self.all_data = self.all_data.sort_values(
                        by=sort_cols, ascending=False, na_position='last'
                    )
                except Exception:
                    pass

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

        # Pre-calcola min/max per colonne numeriche (sull'intero dataset filtrato)
        cf_ranges: dict[str, tuple[float, float]] = {}
        if self.conditional_formatting:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        cf_ranges[col] = (float(vals.min()), float(vals.max()))

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
                col_name = df.columns[j]
                text = self._format_value(col_name, val)
                item = NumericTableWidgetItem(text, sort_value=val)
                item.setTextAlignment(Qt.AlignCenter)

                # ---- Colore sfondo ----
                if (
                    self.conditional_formatting
                    and col_name in cf_ranges
                    and isinstance(val, (int, float))
                    and not pd.isna(val)
                ):
                    cell_color = self._get_cf_color(col_name, val, *cf_ranges[col_name])
                    if cell_color:
                        item.setBackground(cell_color)
                elif row_color:
                    item.setBackground(row_color)

                # ---- Grassetto per own_trade ----
                if is_own:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)

                self.table.setItem(i, j, item)

        self.table.setSortingEnabled(True)

    # ==========================================================
    # CONDITIONAL FORMATTING
    # ==========================================================
    def _toggle_cf(self, checked: bool):
        """Attiva/disattiva il conditional formatting per colonne numeriche."""
        self.conditional_formatting = checked
        self.cf_btn.setText("CF: On" if checked else "CF: Off")
        self.cf_btn.setStyleSheet(
            "background-color: #b2dfdb; font-weight: bold;" if checked else ""
        )
        self._refresh_view()

    def _get_cf_color(
        self, col_name: str, val: float, col_min: float, col_max: float
    ) -> Optional[QColor]:
        """
        Restituisce il QColor per il conditional formatting di una cella.

        • Colonne "con segno" (pnl, profit, return …):
          scala divergente  rosso (min) → bianco (0) → verde (max)
        • Altre colonne numeriche:
          scala sequenziale bianco (min) → blu chiaro (max)
        """
        lower = col_name.lower()
        is_signed = any(kw in lower for kw in _CF_SIGNED_KEYWORDS)

        if is_signed:
            if val > 0 and col_max > 0:
                intensity = min(val / col_max, 1.0)
                green = int(255 - 55 * intensity)   # 255 → 200
                return QColor(200, green, 200)
            elif val < 0 and col_min < 0:
                intensity = min(abs(val) / abs(col_min), 1.0)
                red = int(255 - 55 * intensity)     # 255 → 200
                return QColor(red, 200, 200)
            return None  # zero → nessun colore

        else:
            # Scala sequenziale: bianco (basso) → blu chiaro (alto)
            if col_max > col_min:
                norm = (val - col_min) / (col_max - col_min)
                r = int(255 - norm * 60)
                g = int(255 - norm * 40)
                return QColor(r, g, 255)
            return None

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
