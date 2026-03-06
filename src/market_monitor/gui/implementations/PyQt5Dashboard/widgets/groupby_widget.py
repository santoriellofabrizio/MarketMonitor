"""
Widget GroupBy con multi-colonna aggregazione, filtri avanzati e filtro rolling temporale.
Supporta colonne calcolate passate dal dashboard (già presenti nel DataFrame sorgente).
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QComboBox,
                             QGroupBox, QHeaderView, QFileDialog, QMessageBox,
                             QCheckBox, QMenu, QAction, QWidgetAction, QSpinBox,
                             QDialog, QDialogButtonBox, QLineEdit, QFormLayout,
                             QAbstractItemView)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QCursor
import pandas as pd
from datetime import timedelta
from typing import Optional, Dict, Any, List, Tuple

from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup


class NumericTableWidgetItem(QTableWidgetItem):
    """TableWidgetItem che ordina correttamente i valori numerici"""

    def __lt__(self, other):
        self_value = self.data(Qt.UserRole)
        other_value = other.data(Qt.UserRole)
        if self_value is not None and other_value is not None:
            return self_value < other_value
        return super().__lt__(other)


# ---------------------------------------------------------------------------
# Dialog: seleziona colonna + funzione di aggregazione
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Custom aggregation functions
# ---------------------------------------------------------------------------

def _agg_sum_abs(x):
    """Sum of absolute values."""
    return x.abs().sum()

def _agg_mean_abs(x):
    """Mean of absolute values."""
    return x.abs().mean()

def _agg_diff_abs(x):
    """Mean of absolute consecutive differences (avg step size)."""
    return x.diff().abs().mean()

def _agg_sum_diff_abs(x):
    """Sum of absolute consecutive differences (total variation)."""
    return x.diff().abs().sum()

def _agg_range(x):
    """Range: max - min."""
    return x.max() - x.min()

def _agg_variance(x):
    """Variance."""
    return x.var()

def _agg_first(x):
    """First value in group."""
    return x.iloc[0] if len(x) > 0 else float('nan')

def _agg_last(x):
    """Last value in group."""
    return x.iloc[-1] if len(x) > 0 else float('nan')

def _agg_cv(x):
    """Coefficient of variation (std / mean * 100)."""
    m = x.mean()
    return (x.std() / m * 100) if m != 0 else float('nan')


# Maps display name -> callable (for custom functions) or str (for pandas builtins)
CUSTOM_AGG_MAP = {
    'sum_abs':      _agg_sum_abs,
    'mean_abs':     _agg_mean_abs,
    'diff_abs':     _agg_diff_abs,
    'sum_diff_abs': _agg_sum_diff_abs,
    'range':        _agg_range,
    'variance':     _agg_variance,
    'first':        _agg_first,
    'last':         _agg_last,
    'cv%':          _agg_cv,
}


class AddValueAggDialog(QDialog):
    """Dialog per aggiungere una coppia (colonna, aggregazione) al GroupBy."""

    AGG_FUNCTIONS = [
        # Standard pandas
        'sum', 'mean', 'count', 'min', 'max', 'std', 'median',
        # Custom
        'sum_abs', 'mean_abs', 'diff_abs', 'sum_diff_abs',
        'range', 'variance', 'first', 'last', 'cv%',
    ]

    def __init__(self, available_columns: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Value Column")
        self.setModal(True)
        self.resize(350, 150)
        self._setup_ui(available_columns)

    def _setup_ui(self, available_columns: List[str]):
        layout = QFormLayout(self)

        self.col_combo = QComboBox()
        self.col_combo.addItems(available_columns)
        layout.addRow("Column:", self.col_combo)

        self.agg_combo = QComboBox()
        self.agg_combo.addItems(self.AGG_FUNCTIONS)
        layout.addRow("Aggregation:", self.agg_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_pair(self) -> Tuple[str, str]:
        return self.col_combo.currentText(), self.agg_combo.currentText()


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class GroupByWidget(QWidget):
    """
    Widget per creare e visualizzare group-by con aggregazione multi-colonna.
    Supporta:
    - GroupBy su una colonna (rows)
    - Più colonne valore ognuna con la propria funzione di aggregazione
    - Filtri avanzati con AND/OR
    - Filtro rolling temporale
    - Controllo decimali per colonna (right-click header)
    - Configurazione persistente
    - Sorting numerico corretto
    - Hide/Show settings
    """

    groupby_updated = pyqtSignal(pd.DataFrame)

    TIME_FILTER_OPTIONS = {
        'All': None,
        '1 min': timedelta(minutes=1),
        '5 min': timedelta(minutes=5),
        '15 min': timedelta(minutes=15),
        '30 min': timedelta(minutes=30),
        '1 ora': timedelta(hours=1),
        '4 ore': timedelta(hours=4),
        'Oggi': 'today',
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self.source_data = pd.DataFrame()
        self.result_data = pd.DataFrame()
        self.current_config = None
        self.settings_visible = True

        self.column_decimals: Dict[str, int] = {}
        self.active_filter: Optional[FilterGroup] = None
        self._pending_config: Optional[Dict[str, Any]] = None

        # Coppie (colonna, funzione_aggregazione)
        self.value_agg_pairs: List[Tuple[str, str]] = []

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toggle settings button
        toggle_layout = QHBoxLayout()
        self.toggle_settings_btn = QPushButton("▼ Hide Settings")
        self.toggle_settings_btn.clicked.connect(self._toggle_settings)
        self.toggle_settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        toggle_layout.addWidget(self.toggle_settings_btn)
        toggle_layout.addStretch()
        layout.addLayout(toggle_layout)

        # Settings container (nascondibile)
        self.settings_container = QWidget()
        settings_layout = QVBoxLayout(self.settings_container)
        settings_layout.setContentsMargins(0, 5, 0, 5)

        # --- Filter panel ---
        filter_group = QGroupBox("Data Filters (Applied Before GroupBy)")
        filter_layout = QVBoxLayout()

        filter_row = QHBoxLayout()
        self.advanced_filter_btn = QPushButton("🔍 Advanced Filter")
        self.advanced_filter_btn.clicked.connect(self._show_advanced_filter_dialog)
        self.advanced_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc; color: white;
                font-weight: bold; padding: 5px 15px; border-radius: 3px;
            }
            QPushButton:hover { background-color: #0052a3; }
        """)
        filter_row.addWidget(self.advanced_filter_btn)

        self.clear_filters_btn = QPushButton("Clear Filters")
        self.clear_filters_btn.clicked.connect(self._clear_filters)
        filter_row.addWidget(self.clear_filters_btn)
        filter_row.addStretch()
        filter_layout.addLayout(filter_row)

        self.active_filters_label = QLabel("No filters active")
        self.active_filters_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.active_filters_label.setWordWrap(True)
        filter_layout.addWidget(self.active_filters_label)

        quick_filter_row = QHBoxLayout()
        quick_filter_row.addWidget(QLabel("Time Filter:"))
        self.quick_time_filter = QComboBox()
        self.quick_time_filter.addItems(list(self.TIME_FILTER_OPTIONS.keys()))
        self.quick_time_filter.currentTextChanged.connect(self._on_quick_filter_changed)
        self.quick_time_filter.setMaximumWidth(100)
        quick_filter_row.addWidget(self.quick_time_filter)
        quick_filter_row.addStretch()
        filter_layout.addLayout(quick_filter_row)

        filter_group.setLayout(filter_layout)
        settings_layout.addWidget(filter_group)

        # --- GroupBy configuration panel ---
        config_group = QGroupBox("GroupBy Configuration")
        config_layout = QVBoxLayout()

        # Row: Group By (Rows)
        rows_row = QHBoxLayout()
        rows_row.addWidget(QLabel("Group By (Rows):"))
        self.rows_combo = QComboBox()
        self.rows_combo.addItems(['None'])
        rows_row.addWidget(self.rows_combo)
        rows_row.addStretch()
        config_layout.addLayout(rows_row)

        # Value columns & aggregations table
        config_layout.addWidget(QLabel("Value Columns & Aggregations:"))

        self.value_agg_table = QTableWidget()
        self.value_agg_table.setColumnCount(2)
        self.value_agg_table.setHorizontalHeaderLabels(["Column", "Aggregation"])
        self.value_agg_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.value_agg_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.value_agg_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.value_agg_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.value_agg_table.setMaximumHeight(130)
        self.value_agg_table.setAlternatingRowColors(True)
        config_layout.addWidget(self.value_agg_table)

        # Add / Remove buttons for value_agg_pairs
        va_btn_row = QHBoxLayout()
        add_va_btn = QPushButton("Add Value Column")
        add_va_btn.clicked.connect(self._add_value_agg)
        va_btn_row.addWidget(add_va_btn)

        remove_va_btn = QPushButton("Remove Selected")
        remove_va_btn.clicked.connect(self._remove_selected_value_agg)
        va_btn_row.addWidget(remove_va_btn)
        va_btn_row.addStretch()
        config_layout.addLayout(va_btn_row)

        # Apply / Clear / Export buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply GroupBy")
        self.apply_btn.clicked.connect(self.apply_groupby)
        btn_layout.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_groupby)
        btn_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export Excel")
        self.export_btn.clicked.connect(self._export_excel)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        config_layout.addLayout(btn_layout)

        config_group.setLayout(config_layout)
        settings_layout.addWidget(config_group)

        layout.addWidget(self.settings_container)

        # Result table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._show_header_context_menu)
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #c0c0c0;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.table)

        self.info_label = QLabel("Configure GroupBy and click 'Apply GroupBy'. Right-click headers for decimals.")
        self.info_label.setStyleSheet("font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)

    # ------------------------------------------------------------------
    # Settings toggle
    # ------------------------------------------------------------------

    def _toggle_settings(self):
        self.settings_visible = not self.settings_visible
        if self.settings_visible:
            self.settings_container.show()
            self.toggle_settings_btn.setText("▼ Hide Settings")
        else:
            self.settings_container.hide()
            self.toggle_settings_btn.setText("▶ Show Settings")

    # ------------------------------------------------------------------
    # Value Agg Pairs management
    # ------------------------------------------------------------------

    def _refresh_value_agg_table(self):
        """Sincronizza self.value_agg_pairs → QTableWidget."""
        self.value_agg_table.setRowCount(len(self.value_agg_pairs))
        for row, (col, agg) in enumerate(self.value_agg_pairs):
            self.value_agg_table.setItem(row, 0, QTableWidgetItem(col))
            self.value_agg_table.setItem(row, 1, QTableWidgetItem(agg))

    def _add_value_agg(self):
        """Apre dialog per aggiungere una coppia (colonna, agg)."""
        if self.source_data.empty:
            QMessageBox.warning(self, "No Data", "No source data available yet.")
            return

        numeric_cols = list(self.source_data.select_dtypes(include=['number']).columns)
        if not numeric_cols:
            QMessageBox.warning(self, "No Columns", "No numeric columns available.")
            return

        dialog = AddValueAggDialog(numeric_cols, self)
        if dialog.exec_() == QDialog.Accepted:
            col, agg = dialog.get_pair()
            self.value_agg_pairs.append((col, agg))
            self._refresh_value_agg_table()

    def _remove_selected_value_agg(self):
        """Rimuove le righe selezionate da value_agg_pairs."""
        selected_rows = sorted(
            set(idx.row() for idx in self.value_agg_table.selectedIndexes()),
            reverse=True
        )
        for row in selected_rows:
            if 0 <= row < len(self.value_agg_pairs):
                self.value_agg_pairs.pop(row)
        self._refresh_value_agg_table()

    # ------------------------------------------------------------------
    # Header context menu (decimals)
    # ------------------------------------------------------------------

    def _show_header_context_menu(self, position: QPoint):
        col_index = self.table.horizontalHeader().logicalIndexAt(position)
        if col_index < 0 or col_index >= self.table.columnCount():
            return

        column_name = self.table.horizontalHeaderItem(col_index).text()

        if self.result_data.empty or column_name not in self.result_data.columns:
            return

        col_data = self.result_data[column_name]
        if not pd.api.types.is_numeric_dtype(col_data):
            return

        menu = QMenu(self)

        header_action = QAction(f"📊 Format: {column_name}", self)
        header_action.setEnabled(False)
        font = header_action.font()
        font.setBold(True)
        header_action.setFont(font)
        menu.addAction(header_action)
        menu.addSeparator()

        spinbox_widget = QWidget()
        spinbox_layout = QHBoxLayout(spinbox_widget)
        spinbox_layout.setContentsMargins(10, 5, 10, 5)
        spinbox_layout.addWidget(QLabel("Decimals:"))

        spinbox = QSpinBox()
        spinbox.setRange(0, 6)
        spinbox.setValue(self.column_decimals.get(column_name, 2))
        spinbox.setMinimumWidth(80)
        spinbox_layout.addWidget(spinbox)

        spinbox_action = QWidgetAction(self)
        spinbox_action.setDefaultWidget(spinbox_widget)
        menu.addAction(spinbox_action)
        menu.addSeparator()

        reset_action = QAction("Reset to default (2)", self)
        reset_action.triggered.connect(lambda: self._set_column_decimals(column_name, 2))
        menu.addAction(reset_action)

        spinbox.valueChanged.connect(
            lambda value, c=column_name: self._set_column_decimals(c, value)
        )
        menu.exec_(QCursor.pos())

    def _set_column_decimals(self, column_name: str, decimals: int):
        self.column_decimals[column_name] = decimals
        if not self.result_data.empty:
            self._populate_table()

    # ------------------------------------------------------------------
    # Advanced filter
    # ------------------------------------------------------------------

    def _show_advanced_filter_dialog(self):
        if self.source_data.empty:
            return

        dialog = AdvancedFilterDialog(
            columns=list(self.source_data.columns),
            data=self.source_data,
            current_filter=self.active_filter,
            parent=self
        )
        if dialog.exec_():
            self.active_filter = dialog.get_filter()
            self._update_filter_label()
            if self.current_config:
                self._apply_groupby_from_config(self.current_config)

    def _clear_filters(self):
        self.active_filter = None
        self._update_filter_label()
        if self.current_config:
            self._apply_groupby_from_config(self.current_config)

    def _update_filter_label(self):
        if not self.active_filter or not self.active_filter.conditions:
            self.active_filters_label.setText("No filters active")
            self.active_filters_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
            self.advanced_filter_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0066cc; color: white;
                    font-weight: bold; padding: 5px 15px; border-radius: 3px;
                }
                QPushButton:hover { background-color: #0052a3; }
            """)
        else:
            filter_str = str(self.active_filter)
            num_conditions = len(self.active_filter.conditions)
            if len(filter_str) > 100:
                filter_str = filter_str[:97] + "..."
            self.active_filters_label.setText(
                f"🔍 Active: {num_conditions} condition(s) - {filter_str}"
            )
            self.active_filters_label.setStyleSheet(
                "font-size: 11px; color: #0066cc; font-weight: bold; padding: 5px;"
            )
            self.advanced_filter_btn.setStyleSheet("""
                QPushButton {
                    background-color: #00aa00; color: white;
                    font-weight: bold; padding: 5px 15px; border-radius: 3px;
                }
                QPushButton:hover { background-color: #008800; }
            """)

    def _apply_filters_to_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.active_filter or not self.active_filter.conditions:
            return df
        mask = self.active_filter.apply(df)
        return df[mask]

    # ------------------------------------------------------------------
    # Time filter
    # ------------------------------------------------------------------

    def _on_quick_filter_changed(self):
        if self.current_config:
            self._apply_groupby_from_config(self.current_config)

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ['timestamp', 'time', 'datetime', 'date', 'created_at']
        for col in candidates:
            if col in df.columns:
                return col
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None

    def _apply_time_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        time_filter = self.quick_time_filter.currentText()
        time_delta = self.TIME_FILTER_OPTIONS.get(time_filter)
        if time_delta is None:
            return df

        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col:
            return df

        result = df.copy()
        now = pd.Timestamp.now()
        if time_delta == 'today':
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            cutoff = now - time_delta

        if result[timestamp_col].dtype != 'datetime64[ns]':
            result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors='coerce')

        return result[result[timestamp_col] >= cutoff]

    # ------------------------------------------------------------------
    # Source data
    # ------------------------------------------------------------------

    def set_source_data(self, df: pd.DataFrame):
        """Imposta dati sorgente (già arricchiti con campi calcolati dal dashboard)."""
        self.source_data = df

        if df.empty:
            return

        all_columns = ['None'] + list(df.columns)

        # Preserva selezione corrente
        if self._pending_config:
            current_rows = self._pending_config.get('rows', 'None')
        else:
            current_rows = self.rows_combo.currentText()

        self.rows_combo.clear()
        self.rows_combo.addItems(all_columns)

        if current_rows in all_columns:
            self.rows_combo.setCurrentText(current_rows)

        # Applica pending config se presente
        if self._pending_config:
            self._apply_pending_config()
            self._pending_config = None
        elif self.current_config:
            self._apply_groupby_from_config(self.current_config)

    def _apply_pending_config(self):
        config = self._pending_config
        if not config:
            return

        if 'rows' in config and config['rows']:
            idx = self.rows_combo.findText(config['rows'])
            if idx >= 0:
                self.rows_combo.setCurrentIndex(idx)

        if 'value_agg_pairs' in config:
            self.value_agg_pairs = [tuple(p) for p in config['value_agg_pairs']]
            self._refresh_value_agg_table()

        if 'quick_time_filter' in config:
            idx = self.quick_time_filter.findText(config['quick_time_filter'])
            if idx >= 0:
                self.quick_time_filter.setCurrentIndex(idx)

        if 'settings_visible' in config:
            self.settings_visible = config['settings_visible']
            if self.settings_visible:
                self.settings_container.show()
                self.toggle_settings_btn.setText("▼ Hide Settings")
            else:
                self.settings_container.hide()
                self.toggle_settings_btn.setText("▶ Show Settings")

        if config.get('current_config'):
            self.current_config = config['current_config']
            self._apply_groupby_from_config(self.current_config)

    # ------------------------------------------------------------------
    # GroupBy application
    # ------------------------------------------------------------------

    def apply_groupby(self):
        if self.source_data.empty:
            QMessageBox.warning(self, "No Data", "No source data available")
            return

        rows = self.rows_combo.currentText()

        if rows == 'None':
            QMessageBox.warning(self, "Invalid Config", "Please select 'Group By (Rows)'")
            return

        if not self.value_agg_pairs:
            QMessageBox.warning(self, "Invalid Config",
                                "Please add at least one Value Column with an Aggregation function.")
            return

        self.current_config = {
            'rows': rows,
            'value_agg_pairs': list(self.value_agg_pairs),
        }
        self._apply_groupby_from_config(self.current_config)

    def _apply_groupby_from_config(self, config: Dict[str, Any]):
        try:
            rows = config['rows']
            pairs = config['value_agg_pairs']

            # Pipeline filtri
            filtered = self._apply_filters_to_data(self.source_data)
            filtered = self._apply_time_filter(filtered)

            original_count = len(self.source_data)
            filtered_count = len(filtered)

            if filtered.empty:
                self.info_label.setText("No data after filtering")
                self.result_data = pd.DataFrame()
                self._populate_table()
                return

            if rows not in filtered.columns:
                self.info_label.setText(f"Column '{rows}' not found in data")
                return

            # Verifica colonne valore
            missing = [col for col, _ in pairs if col not in filtered.columns]
            if missing:
                self.info_label.setText(f"Columns not found: {missing}")
                return

            # GroupBy multi-aggregazione
            # Resolve custom function names to callables
            agg_dict = {
                col: CUSTOM_AGG_MAP.get(agg, agg)
                for col, agg in pairs
            }
            result = filtered.groupby(rows).agg(agg_dict).reset_index()

            # Rinomina colonne: col -> agg_col
            new_cols = [rows] + [f'{agg}_{col}' for col, agg in pairs]
            result.columns = new_cols

            self.result_data = result
            self._populate_table()
            self.groupby_updated.emit(self.result_data)

            # Info label
            info_parts = []
            pairs_str = ", ".join(f"{agg}({col})" for col, agg in pairs)
            info_parts.append(f"GroupBy: {rows} | {pairs_str}")

            if filtered_count < original_count:
                info_parts.append(f"Data: {filtered_count}/{original_count}")

            time_filter = self.quick_time_filter.currentText()
            if time_filter != 'All':
                info_parts.append(f"Time: {time_filter}")

            info_parts.append(f"{len(self.result_data)} rows")
            self.info_label.setText(" | ".join(info_parts))

        except Exception as e:
            QMessageBox.warning(self, "GroupBy Error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_config = None

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self):
        if self.result_data.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.result_data))
        self.table.setColumnCount(len(self.result_data.columns))
        self.table.setHorizontalHeaderLabels([str(col) for col in self.result_data.columns])

        # Min/max per colorazione
        numeric_data = []
        for j in range(1, len(self.result_data.columns)):
            col_data = self.result_data.iloc[:, j]
            if pd.api.types.is_numeric_dtype(col_data):
                numeric_data.extend([v for v in col_data if pd.notna(v) and v != 0])

        min_val = min(numeric_data) if numeric_data else 0
        max_val = max(numeric_data) if numeric_data else 0

        for i, row in enumerate(self.result_data.itertuples(index=False)):
            for j, value in enumerate(row):
                if isinstance(value, (int, float)):
                    if pd.isna(value):
                        text = ""
                    else:
                        col_name = str(self.result_data.columns[j])
                        decimals = self.column_decimals.get(col_name, 2)
                        text = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        if decimals == 0:
                            text = text.replace(',00', '')
                        elif decimals > 0:
                            text = text.rstrip('0').rstrip(',')
                else:
                    text = str(value) if not pd.isna(value) else ""

                if isinstance(value, (int, float)) and not pd.isna(value):
                    item = NumericTableWidgetItem(text)
                    item.setData(Qt.UserRole, float(value))
                else:
                    item = QTableWidgetItem(text)

                item.setTextAlignment(Qt.AlignCenter)

                # Colorazione gradiente
                if isinstance(value, (int, float)) and j > 0 and not pd.isna(value) and value != 0:
                    if value > 0 and max_val > 0:
                        intensity = min(abs(value) / max_val, 1.0)
                        green = int(255 - 55 * intensity)
                        item.setBackground(QColor(200, green, 200))
                    elif value > 0:
                        item.setBackground(QColor(200, 255, 200))
                    elif value < 0 and min_val < 0:
                        intensity = min(abs(value) / abs(min_val), 1.0)
                        red = int(255 - 55 * intensity)
                        item.setBackground(QColor(red, 200, 200))
                    elif value < 0:
                        item.setBackground(QColor(255, 200, 200))

                self.table.setItem(i, j, item)

        header = self.table.horizontalHeader()
        for i in range(len(self.result_data.columns)):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
            self.table.setColumnWidth(i, 120 if i == 0 else 100)

        self.table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # Clear / Export
    # ------------------------------------------------------------------

    def clear_groupby(self):
        self.current_config = None
        self.result_data = pd.DataFrame()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.info_label.setText("GroupBy cleared")
        self.groupby_updated.emit(pd.DataFrame())

    def _export_excel(self):
        if self.result_data.empty:
            QMessageBox.warning(self, "Export", "No data to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Excel",
            f"groupby_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel Files (*.xlsx)"
        )
        if filename:
            try:
                self.result_data.to_excel(filename, index=False)
                QMessageBox.information(self, "Export", f"Exported to {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            'rows': self.rows_combo.currentText(),
            'value_agg_pairs': [list(p) for p in self.value_agg_pairs],
            'settings_visible': self.settings_visible,
            'current_config': (
                {
                    'rows': self.current_config['rows'],
                    'value_agg_pairs': [list(p) for p in self.current_config['value_agg_pairs']],
                }
                if self.current_config else None
            ),
            'column_decimals': self.column_decimals.copy(),
            'quick_time_filter': self.quick_time_filter.currentText(),
        }

    def restore_config(self, config: dict):
        try:
            if 'column_decimals' in config:
                self.column_decimals = config['column_decimals'].copy()

            if 'settings_visible' in config:
                self.settings_visible = config['settings_visible']
                if self.settings_visible:
                    self.settings_container.show()
                    self.toggle_settings_btn.setText("▼ Hide Settings")
                else:
                    self.settings_container.hide()
                    self.toggle_settings_btn.setText("▶ Show Settings")

            if self.source_data.empty:
                self._pending_config = config.copy()
                return

            if 'rows' in config and config['rows']:
                idx = self.rows_combo.findText(config['rows'])
                if idx >= 0:
                    self.rows_combo.setCurrentIndex(idx)

            if 'value_agg_pairs' in config:
                self.value_agg_pairs = [tuple(p) for p in config['value_agg_pairs']]
                self._refresh_value_agg_table()

            if 'quick_time_filter' in config:
                idx = self.quick_time_filter.findText(config['quick_time_filter'])
                if idx >= 0:
                    self.quick_time_filter.setCurrentIndex(idx)

            if config.get('current_config'):
                self.current_config = {
                    'rows': config['current_config']['rows'],
                    'value_agg_pairs': [tuple(p) for p in config['current_config']['value_agg_pairs']],
                }
                self._apply_groupby_from_config(self.current_config)

        except Exception as e:
            print(f"Error restoring groupby config: {e}")
            import traceback
            traceback.print_exc()

    def get_result_data(self) -> pd.DataFrame:
        return self.result_data.copy() if not self.result_data.empty else pd.DataFrame()
