"""
Widget Pivot Table con filtri avanzati AND/OR, controllo decimali e filtro rolling temporale.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QComboBox,
                             QGroupBox, QHeaderView, QFileDialog, QMessageBox,
                             QCheckBox, QMenu, QAction, QWidgetAction, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QCursor
import pandas as pd
from datetime import timedelta
from typing import Optional, Dict, Any, List

# Import sistema filtri avanzati
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup


class NumericTableWidgetItem(QTableWidgetItem):
    """TableWidgetItem che ordina correttamente i valori numerici"""

    def __lt__(self, other):
        self_value = self.data(Qt.UserRole)
        other_value = other.data(Qt.UserRole)

        if self_value is not None and other_value is not None:
            return self_value < other_value

        return super().__lt__(other)


class PivotTableWidget(QWidget):
    """
    Widget per creare e visualizzare pivot tables.
    Supporta:
    - Groupby e pivot
    - Filtri avanzati con AND/OR
    - Filtro rolling temporale
    - Normalizzazione (rows/cols/all)
    - Controllo decimali per colonna
    - Configurazione persistente
    - Sorting numerico corretto
    - Hide/Show settings
    """

    pivot_updated = pyqtSignal(pd.DataFrame)

    # Opzioni filtro temporale rapido (stesso formato di ChartWidget)
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
        self.pivot_data = pd.DataFrame()
        self.current_config = None
        self.settings_visible = True

        # Filtri avanzati e decimali
        self.column_decimals = {}
        self.active_filter: Optional[FilterGroup] = None

        # Configurazione pending per restore dopo set_source_data
        self._pending_config: Optional[Dict[str, Any]] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup interfaccia"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toggle Settings Button
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
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        toggle_layout.addWidget(self.toggle_settings_btn)
        toggle_layout.addStretch()

        layout.addLayout(toggle_layout)

        # Settings Container (nascondibile)
        self.settings_container = QWidget()
        settings_layout = QVBoxLayout(self.settings_container)
        settings_layout.setContentsMargins(0, 5, 0, 5)

        # Filter Panel
        filter_group = QGroupBox("Data Filters (Applied Before Pivot)")
        filter_layout = QVBoxLayout()

        filter_row = QHBoxLayout()

        # Bottone filtro avanzato
        self.advanced_filter_btn = QPushButton("🔍 Advanced Filter")
        self.advanced_filter_btn.clicked.connect(self._show_advanced_filter_dialog)
        self.advanced_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
        """)
        filter_row.addWidget(self.advanced_filter_btn)

        self.clear_filters_btn = QPushButton("Clear Filters")
        self.clear_filters_btn.clicked.connect(self._clear_filters)
        filter_row.addWidget(self.clear_filters_btn)

        filter_row.addStretch()
        filter_layout.addLayout(filter_row)

        # Active filters label
        self.active_filters_label = QLabel("No filters active")
        self.active_filters_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.active_filters_label.setWordWrap(True)
        filter_layout.addWidget(self.active_filters_label)

        # Filtro temporale rapido (rolling)
        quick_filter_row = QHBoxLayout()

        quick_filter_row.addWidget(QLabel("Time Filter:"))
        self.quick_time_filter = QComboBox()
        self.quick_time_filter.addItems(list(self.TIME_FILTER_OPTIONS.keys()))
        self.quick_time_filter.setToolTip("Filtra i dati per finestra temporale rolling prima del pivot")
        self.quick_time_filter.currentTextChanged.connect(self._on_quick_filter_changed)
        self.quick_time_filter.setMaximumWidth(100)
        quick_filter_row.addWidget(self.quick_time_filter)

        quick_filter_row.addStretch()
        filter_layout.addLayout(quick_filter_row)

        filter_group.setLayout(filter_layout)
        settings_layout.addWidget(filter_group)

        # Configuration Panel
        config_group = QGroupBox("Pivot Configuration")
        config_layout = QVBoxLayout()

        # Row 1: Rows and Columns
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Group By (Rows):"))
        self.rows_combo = QComboBox()
        self.rows_combo.addItems(['None'])
        row1.addWidget(self.rows_combo)

        row1.addWidget(QLabel("Columns:"))
        self.cols_combo = QComboBox()
        self.cols_combo.addItems(['None'])
        row1.addWidget(self.cols_combo)

        config_layout.addLayout(row1)

        # Row 2: Values and Aggregation
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Values:"))
        self.values_combo = QComboBox()
        self.values_combo.addItems(['None'])
        row2.addWidget(self.values_combo)

        row2.addWidget(QLabel("Aggregation:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(['sum', 'mean', 'count', 'min', 'max', 'std', 'median'])
        row2.addWidget(self.agg_combo)

        config_layout.addLayout(row2)

        # Row 3: Normalization options
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Normalize:"))

        self.normalize_none_radio = QCheckBox("None")
        self.normalize_none_radio.setChecked(True)
        row3.addWidget(self.normalize_none_radio)

        self.normalize_rows_radio = QCheckBox("By Rows (%)")
        row3.addWidget(self.normalize_rows_radio)

        self.normalize_cols_radio = QCheckBox("By Columns (%)")
        row3.addWidget(self.normalize_cols_radio)

        self.normalize_all_radio = QCheckBox("By Total (%)")
        row3.addWidget(self.normalize_all_radio)

        # Make checkboxes mutually exclusive
        for checkbox in [self.normalize_none_radio, self.normalize_rows_radio,
                         self.normalize_cols_radio, self.normalize_all_radio]:
            checkbox.stateChanged.connect(
                lambda state, cb=checkbox: self._on_normalize_changed(cb)
            )

        row3.addStretch()
        config_layout.addLayout(row3)

        # Buttons
        btn_layout = QHBoxLayout()

        self.apply_btn = QPushButton("Apply Pivot")
        self.apply_btn.clicked.connect(self.apply_pivot)
        btn_layout.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("Clear Pivot")
        self.clear_btn.clicked.connect(self.clear_pivot)
        btn_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export Excel")
        self.export_btn.clicked.connect(self._export_excel)
        btn_layout.addWidget(self.export_btn)

        btn_layout.addStretch()

        config_layout.addLayout(btn_layout)
        config_group.setLayout(config_layout)
        settings_layout.addWidget(config_group)

        layout.addWidget(self.settings_container)

        # Pivot Table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

        # Menu contestuale su header per decimali
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._show_header_context_menu)

        # Style
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

        # Info label
        self.info_label = QLabel("Configure pivot and click 'Apply Pivot'. Right-click headers for decimals.")
        self.info_label.setStyleSheet("font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)

    def _toggle_settings(self):
        """Toggle visibilità dei settings"""
        self.settings_visible = not self.settings_visible

        if self.settings_visible:
            self.settings_container.show()
            self.toggle_settings_btn.setText("▼ Hide Settings")
        else:
            self.settings_container.hide()
            self.toggle_settings_btn.setText("▶ Show Settings")

    def _show_header_context_menu(self, position: QPoint):
        """Menu contestuale su header colonna per decimali"""
        col_index = self.table.horizontalHeader().logicalIndexAt(position)
        if col_index < 0 or col_index >= self.table.columnCount():
            return

        column_name = self.table.horizontalHeaderItem(col_index).text()

        if self.pivot_data.empty or column_name not in self.pivot_data.columns:
            return

        col_data = self.pivot_data[column_name]
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
        current_decimals = self.column_decimals.get(column_name, 2)
        spinbox.setValue(current_decimals)
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
        """Imposta decimali per una colonna"""
        self.column_decimals[column_name] = decimals
        if not self.pivot_data.empty:
            self._populate_table()

    def _show_advanced_filter_dialog(self):
        """Mostra dialog filtro avanzato"""
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
                self._apply_pivot_from_config(self.current_config)

    def _clear_filters(self):
        """Pulisce tutti i filtri"""
        self.active_filter = None
        self._update_filter_label()
        if self.current_config:
            self._apply_pivot_from_config(self.current_config)

    def _update_filter_label(self):
        """Aggiorna label filtri attivi"""
        if not self.active_filter or not self.active_filter.conditions:
            self.active_filters_label.setText("No filters active")
            self.active_filters_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
            self.advanced_filter_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0066cc;
                    color: white;
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #0052a3;
                }
            """)
        else:
            filter_str = str(self.active_filter)
            num_conditions = len(self.active_filter.conditions)

            if len(filter_str) > 100:
                filter_str = filter_str[:97] + "..."

            self.active_filters_label.setText(
                f"🔍 Active: {num_conditions} condition(s) - {filter_str}"
            )
            self.active_filters_label.setStyleSheet("font-size: 11px; color: #0066cc; font-weight: bold; padding: 5px;")
            self.advanced_filter_btn.setStyleSheet("""
                QPushButton {
                    background-color: #00aa00;
                    color: white;
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #008800;
                }
            """)

    def _apply_filters_to_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica filtri avanzati al DataFrame"""
        if not self.active_filter or not self.active_filter.conditions:
            return df

        mask = self.active_filter.apply(df)
        return df[mask]

    def _on_quick_filter_changed(self):
        """Callback quando cambia il filtro temporale rapido."""
        if self.current_config:
            self._apply_pivot_from_config(self.current_config)

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Trova la colonna timestamp nel DataFrame."""
        candidates = ['timestamp', 'time', 'datetime', 'date', 'created_at']

        for col in candidates:
            if col in df.columns:
                return col

        # Cerca colonne di tipo datetime
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _apply_time_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica filtro temporale rolling.
        Riduce i dati "alla fonte" per alleggerire il pivot.
        """
        if df.empty:
            return df

        time_filter = self.quick_time_filter.currentText()
        time_delta = self.TIME_FILTER_OPTIONS.get(time_filter)

        if time_delta is None:
            return df

        # Trova colonna timestamp
        timestamp_col = self._find_timestamp_column(df)

        if not timestamp_col:
            return df

        result = df.copy()
        now = pd.Timestamp.now()

        if time_delta == 'today':
            # Filtra per "oggi"
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            cutoff = now - time_delta

        # Assicurati che la colonna sia datetime
        if result[timestamp_col].dtype != 'datetime64[ns]':
            result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors='coerce')

        result = result[result[timestamp_col] >= cutoff]

        return result

    def _on_normalize_changed(self, changed_checkbox):
        """Gestisce mutually exclusive checkboxes per normalize"""
        if changed_checkbox.isChecked():
            for checkbox in [self.normalize_none_radio, self.normalize_rows_radio,
                             self.normalize_cols_radio, self.normalize_all_radio]:
                if checkbox != changed_checkbox:
                    checkbox.setChecked(False)

    def set_source_data(self, df: pd.DataFrame):
        """Imposta i dati sorgente per il pivot"""
        self.source_data = df

        if df.empty:
            return

        # Aggiorna combo box
        columns = ['None'] + list(df.columns)

        # Salva selezione corrente (o da pending config)
        if self._pending_config:
            current_rows = self._pending_config.get('rows', 'None')
            current_cols = self._pending_config.get('cols', 'None')
            current_values = self._pending_config.get('values', 'None')
        else:
            current_rows = self.rows_combo.currentText()
            current_cols = self.cols_combo.currentText()
            current_values = self.values_combo.currentText()

        # Aggiorna combo boxes
        self.rows_combo.clear()
        self.rows_combo.addItems(columns)

        self.cols_combo.clear()
        self.cols_combo.addItems(columns)

        # Solo colonne numeriche per values
        numeric_cols = ['None'] + list(df.select_dtypes(include=['number']).columns)
        self.values_combo.clear()
        self.values_combo.addItems(numeric_cols)

        # Ripristina selezione se possibile
        if current_rows in columns:
            self.rows_combo.setCurrentText(current_rows)
        if current_cols in columns:
            self.cols_combo.setCurrentText(current_cols)
        if current_values in numeric_cols:
            self.values_combo.setCurrentText(current_values)

        # Applica pending config se presente
        if self._pending_config:
            self._apply_pending_config()
            self._pending_config = None
        # Altrimenti ri-applica pivot se configurato
        elif self.current_config:
            self._apply_pivot_from_config(self.current_config)

    def _apply_pending_config(self):
        """Applica la configurazione pending dopo che le combo box sono state popolate."""
        config = self._pending_config
        if not config:
            return

        # Applica selezioni combo box
        if 'rows' in config and config['rows']:
            index = self.rows_combo.findText(config['rows'])
            if index >= 0:
                self.rows_combo.setCurrentIndex(index)

        if 'cols' in config and config['cols']:
            index = self.cols_combo.findText(config['cols'])
            if index >= 0:
                self.cols_combo.setCurrentIndex(index)

        if 'values' in config and config['values']:
            index = self.values_combo.findText(config['values'])
            if index >= 0:
                self.values_combo.setCurrentIndex(index)

        if 'agg' in config and config['agg']:
            index = self.agg_combo.findText(config['agg'])
            if index >= 0:
                self.agg_combo.setCurrentIndex(index)

        # Applica normalize
        if config.get('normalize_none', True):
            self.normalize_none_radio.setChecked(True)
        elif config.get('normalize_rows', False):
            self.normalize_rows_radio.setChecked(True)
        elif config.get('normalize_cols', False):
            self.normalize_cols_radio.setChecked(True)
        elif config.get('normalize_all', False):
            self.normalize_all_radio.setChecked(True)

        # Applica filtro temporale
        if 'quick_time_filter' in config:
            index = self.quick_time_filter.findText(config['quick_time_filter'])
            if index >= 0:
                self.quick_time_filter.setCurrentIndex(index)

        # Applica current_config per il pivot
        if config.get('current_config'):
            self.current_config = config['current_config']
            self._apply_pivot_from_config(self.current_config)

    def apply_pivot(self):
        """Applica la configurazione pivot"""
        if self.source_data.empty:
            QMessageBox.warning(self, "No Data", "No source data available")
            return

        rows = self.rows_combo.currentText()
        cols = self.cols_combo.currentText()
        values = self.values_combo.currentText()
        agg = self.agg_combo.currentText()

        # Determina normalizzazione
        normalize = None
        if self.normalize_rows_radio.isChecked():
            normalize = 'index'
        elif self.normalize_cols_radio.isChecked():
            normalize = 'columns'
        elif self.normalize_all_radio.isChecked():
            normalize = 'all'

        # Validazione
        if rows == 'None':
            QMessageBox.warning(self, "Invalid Config", "Please select 'Group By (Rows)'")
            return

        if values == 'None':
            QMessageBox.warning(self, "Invalid Config", "Please select 'Values'")
            return

        # Salva configurazione
        self.current_config = {
            'rows': rows,
            'cols': cols,
            'values': values,
            'agg': agg,
            'normalize': normalize
        }

        self._apply_pivot_from_config(self.current_config)

    def _apply_pivot_from_config(self, config: Dict[str, Any]):
        """Applica pivot usando configurazione salvata con filtri in cascata."""
        try:
            rows = config['rows']
            cols = config['cols']
            values = config['values']
            agg = config['agg']
            normalize = config.get('normalize', None)

            # Applica filtri in cascata:
            # 1. Filtri avanzati (AND/OR)
            filtered_data = self._apply_filters_to_data(self.source_data)

            # 2. Filtro temporale rolling
            filtered_data = self._apply_time_filter(filtered_data)

            # Salva conteggi per info label
            original_count = len(self.source_data)
            filtered_count = len(filtered_data)

            if filtered_data.empty:
                # Non mostrare warning popup per evitare flood durante auto-update
                self.info_label.setText("No data after filtering")
                self.pivot_data = pd.DataFrame()
                self._populate_table()
                return

            # Verifica colonne
            if rows not in filtered_data.columns:
                self.info_label.setText(f"Column '{rows}' not found in data")
                return

            if values not in filtered_data.columns:
                self.info_label.setText(f"Column '{values}' not found in data")
                return

            if cols == 'None':
                # Simple groupby
                self.pivot_data = filtered_data.groupby(rows)[values].agg(agg).reset_index()
                self.pivot_data.columns = [rows, f'{agg}_{values}']
            else:
                # Full pivot table
                if cols not in filtered_data.columns:
                    self.info_label.setText(f"Column '{cols}' not found in data")
                    return

                self.pivot_data = pd.pivot_table(
                    filtered_data,
                    values=values,
                    index=rows,
                    columns=cols,
                    aggfunc=agg,
                    fill_value=0
                ).reset_index()
                
                # Fix FutureWarning: infer objects dopo fillna
                self.pivot_data = self.pivot_data.infer_objects(copy=False)

            # Normalizzazione
            if normalize:
                self.pivot_data = self._apply_normalization(self.pivot_data, normalize)

            self._populate_table()
            self.pivot_updated.emit(self.pivot_data)

            # Update info con dettagli filtri
            info_parts = []

            config_str = f"{rows} x {cols if cols != 'None' else 'N/A'} - {agg}({values})"
            info_parts.append(f"Pivot: {config_str}")

            if normalize:
                info_parts.append(f"Norm: {normalize}")

            # Mostra conteggio se filtrato
            if filtered_count < original_count:
                info_parts.append(f"Data: {filtered_count}/{original_count}")

            # Mostra filtro temporale attivo
            time_filter = self.quick_time_filter.currentText()
            if time_filter != 'All':
                info_parts.append(f"Time: {time_filter}")

            info_parts.append(f"{len(self.pivot_data)} rows")

            self.info_label.setText(" | ".join(info_parts))

        except Exception as e:
            QMessageBox.warning(self, "Pivot Error", f"Error creating pivot: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_config = None

    def _apply_normalization(self, df: pd.DataFrame, normalize: str) -> pd.DataFrame:
        """Applica normalizzazione al pivot"""
        if df.empty:
            return df

        result = df.copy()

        # Identifica colonne numeriche
        numeric_cols = []
        for col in result.columns[1:]:
            if pd.api.types.is_numeric_dtype(result[col]):
                numeric_cols.append(col)

        if not numeric_cols:
            return df

        if normalize == 'index':
            # Normalize by rows
            for idx in result.index:
                row_sum = result.loc[idx, numeric_cols].sum()
                if row_sum != 0:
                    for col in numeric_cols:
                        result.loc[idx, col] = (result.loc[idx, col] / row_sum) * 100
                else:
                    for col in numeric_cols:
                        result.loc[idx, col] = 0

        elif normalize == 'columns':
            # Normalize by columns
            for col in numeric_cols:
                col_sum = result[col].sum()
                if col_sum != 0:
                    result[col] = (result[col] / col_sum) * 100
                else:
                    result[col] = 0

        elif normalize == 'all':
            # Normalize by total
            total = result[numeric_cols].sum().sum()
            if total != 0:
                for col in numeric_cols:
                    result[col] = (result[col] / total) * 100
            else:
                for col in numeric_cols:
                    result[col] = 0

        return result

    def _populate_table(self):
        """Popola la tabella con i dati pivot"""
        if self.pivot_data.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        self.table.setSortingEnabled(False)

        # Setup
        self.table.setRowCount(len(self.pivot_data))
        self.table.setColumnCount(len(self.pivot_data.columns))
        self.table.setHorizontalHeaderLabels([str(col) for col in self.pivot_data.columns])

        # Determina se percentuali
        is_percentage = (self.normalize_rows_radio.isChecked() or
                         self.normalize_cols_radio.isChecked() or
                         self.normalize_all_radio.isChecked())

        # Min/max per gradiente
        numeric_data = []
        for j in range(1, len(self.pivot_data.columns)):
            col_data = self.pivot_data.iloc[:, j]
            if pd.api.types.is_numeric_dtype(col_data):
                numeric_data.extend([v for v in col_data if pd.notna(v) and v != 0])

        if numeric_data:
            min_val = min(numeric_data)
            max_val = max(numeric_data)
        else:
            min_val = 0
            max_val = 0

        # Populate
        for i, row in enumerate(self.pivot_data.itertuples(index=False)):
            for j, value in enumerate(row):
                if isinstance(value, (int, float)):
                    if pd.isna(value):
                        text = ""
                    else:
                        # USA decimali personalizzati
                        col_name = str(self.pivot_data.columns[j])
                        decimals = self.column_decimals.get(col_name, 2)

                        if is_percentage and j > 0:
                            text = f"{value:.{decimals}f}%"
                        else:
                            text = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
                            if decimals == 0:
                                text = text.replace(',00', '')
                            elif decimals > 0:
                                text = text.rstrip('0').rstrip(',')
                else:
                    text = str(value) if not pd.isna(value) else ""

                # TableWidgetItem
                if isinstance(value, (int, float)) and not pd.isna(value):
                    item = NumericTableWidgetItem(text)
                    item.setData(Qt.UserRole, float(value))
                else:
                    item = QTableWidgetItem(text)

                item.setTextAlignment(Qt.AlignCenter)

                # Colorazione
                if isinstance(value, (int, float)) and j > 0 and not pd.isna(value) and value != 0:
                    if value > 0:
                        if max_val > 0:
                            intensity_factor = min(abs(value) / max_val, 1.0)
                            green = int(255 - (55 * intensity_factor))
                            item.setBackground(QColor(200, green, 200))
                        else:
                            item.setBackground(QColor(200, 255, 200))
                    elif value < 0:
                        if min_val < 0:
                            intensity_factor = min(abs(value) / abs(min_val), 1.0)
                            red = int(255 - (55 * intensity_factor))
                            item.setBackground(QColor(red, 200, 200))
                        else:
                            item.setBackground(QColor(255, 200, 200))

                self.table.setItem(i, j, item)

        # Auto-resize
        header = self.table.horizontalHeader()
        for i in range(len(self.pivot_data.columns)):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
            if i == 0:
                self.table.setColumnWidth(i, 120)
            else:
                self.table.setColumnWidth(i, 100)

        self.table.setSortingEnabled(True)

    def clear_pivot(self):
        """Pulisce il pivot"""
        self.current_config = None
        self.pivot_data = pd.DataFrame()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.info_label.setText("Pivot cleared")
        self.pivot_updated.emit(pd.DataFrame())

        # Reset normalize
        self.normalize_none_radio.setChecked(True)
        self.normalize_rows_radio.setChecked(False)
        self.normalize_cols_radio.setChecked(False)
        self.normalize_all_radio.setChecked(False)

    def _export_excel(self):
        """Esporta il pivot in Excel"""
        if self.pivot_data.empty:
            QMessageBox.warning(self, "Export", "No pivot data to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Excel",
            f"pivot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel Files (*.xlsx)"
        )

        if filename:
            try:
                self.pivot_data.to_excel(filename, index=False)
                QMessageBox.information(self, "Export", f"Pivot exported to {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    def get_pivot_data(self) -> pd.DataFrame:
        """Ritorna i dati pivot correnti"""
        return self.pivot_data.copy() if not self.pivot_data.empty else pd.DataFrame()

    def get_config(self) -> dict:
        """Salva configurazione corrente (include filtro temporale)."""
        return {
            'rows': self.rows_combo.currentText(),
            'cols': self.cols_combo.currentText(),
            'values': self.values_combo.currentText(),
            'agg': self.agg_combo.currentText(),
            'normalize_none': self.normalize_none_radio.isChecked(),
            'normalize_rows': self.normalize_rows_radio.isChecked(),
            'normalize_cols': self.normalize_cols_radio.isChecked(),
            'normalize_all': self.normalize_all_radio.isChecked(),
            'settings_visible': self.settings_visible,
            'current_config': self.current_config.copy() if self.current_config else None,
            'column_decimals': self.column_decimals.copy(),
            # Filtro temporale rapido
            'quick_time_filter': self.quick_time_filter.currentText(),
        }

    def restore_config(self, config: dict):
        """
        Ripristina configurazione salvata.

        Se i dati non sono ancora disponibili (combo box vuote),
        salva la config come pending e la applica in set_source_data.
        """
        try:
            # Salva column_decimals subito (non dipende dai dati)
            if 'column_decimals' in config:
                self.column_decimals = config['column_decimals'].copy()

            # Ripristina settings visibility subito
            if 'settings_visible' in config:
                self.settings_visible = config['settings_visible']
                if self.settings_visible:
                    self.settings_container.show()
                    self.toggle_settings_btn.setText("▼ Hide Settings")
                else:
                    self.settings_container.hide()
                    self.toggle_settings_btn.setText("▶ Show Settings")

            # Se non ci sono dati, salva config come pending
            if self.source_data.empty:
                self._pending_config = config.copy()
                return

            # Altrimenti applica subito
            if 'rows' in config and config['rows']:
                index = self.rows_combo.findText(config['rows'])
                if index >= 0:
                    self.rows_combo.setCurrentIndex(index)

            if 'cols' in config and config['cols']:
                index = self.cols_combo.findText(config['cols'])
                if index >= 0:
                    self.cols_combo.setCurrentIndex(index)

            if 'values' in config and config['values']:
                index = self.values_combo.findText(config['values'])
                if index >= 0:
                    self.values_combo.setCurrentIndex(index)

            if 'agg' in config and config['agg']:
                index = self.agg_combo.findText(config['agg'])
                if index >= 0:
                    self.agg_combo.setCurrentIndex(index)

            if config.get('normalize_none', True):
                self.normalize_none_radio.setChecked(True)
            elif config.get('normalize_rows', False):
                self.normalize_rows_radio.setChecked(True)
            elif config.get('normalize_cols', False):
                self.normalize_cols_radio.setChecked(True)
            elif config.get('normalize_all', False):
                self.normalize_all_radio.setChecked(True)

            # Ripristina filtro temporale rapido
            if 'quick_time_filter' in config:
                index = self.quick_time_filter.findText(config['quick_time_filter'])
                if index >= 0:
                    self.quick_time_filter.setCurrentIndex(index)

            if config.get('current_config'):
                self.current_config = config['current_config']
                self._apply_pivot_from_config(self.current_config)

        except Exception as e:
            print(f"Error restoring pivot config: {e}")
            import traceback
            traceback.print_exc()
