"""
ChartWidget con filtri avanzati AND/OR e ottimizzazioni per stabilità.

Refactoring in 3 fasi:
- Fase 1: Throttling e reattività (debounce, rendering flag, datetime preprocessing)
- Fase 2: Filtro rolling temporale e "Top N" categorie
- Fase 3: Ottimizzazione rendering (downsampling, aggiornamento incrementale)

- Mode: Static Snapshot | Time Evolution
- Filtri avanzati con logica AND/OR
- Auto-update con debounce
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QPushButton, QRadioButton,
                             QSpinBox, QCheckBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import sistema filtri avanzati
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup


class ChartWidget(QWidget):
    """
    Widget versatile per chart con filtri avanzati e dual mode.

    Ottimizzazioni per stabilità:
    - Debounce su set_data per evitare ridisegni multipli
    - Protezione rendering con flag _is_rendering
    - Preprocessing datetime per performance
    - Filtro temporale rapido (rolling)
    - Limitatore "Top N" per bar chart
    - Downsampling intelligente per line chart
    """

    # ========================
    # CONFIGURAZIONE THROTTLING (Fase 1)
    # ========================
    DEBOUNCE_MS = 300  # Millisecondi di attesa prima del render
    MAX_POINTS_LINE_CHART = 2000  # Soglia per downsampling (Fase 3)
    DEFAULT_TOP_N = 20  # Default barre per bar chart (Fase 2)

    # Opzioni filtro temporale rapido (Fase 2)
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

        self.logger = logging.getLogger(__name__)

        self.source_data = pd.DataFrame()
        self.current_config = None
        self.mode = 'static'

        # ========================
        # FASE 1: STATO THROTTLING
        # ========================
        self._is_rendering = False  # Flag protezione render
        self._pending_render = False  # Render in attesa

        # Timer per debounce
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        # Riferimenti per aggiornamento incrementale (Fase 3)
        self._line_objects: Dict[str, Any] = {}
        self._last_chart_mode: Optional[str] = None

        # Filtri avanzati
        self.active_filter: Optional[FilterGroup] = None

        # Configurazione pending per restore dopo set_data
        self._pending_config: Optional[Dict[str, Any]] = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup interfaccia"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toggle Settings
        toggle_row = QHBoxLayout()
        self.toggle_settings_btn = QPushButton("▼ Hide Settings")
        self.toggle_settings_btn.clicked.connect(self._toggle_settings)
        self.toggle_settings_btn.setMaximumWidth(150)
        toggle_row.addWidget(self.toggle_settings_btn)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # Settings Container
        self.settings_container = QWidget()
        settings_layout = QVBoxLayout(self.settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        # Filter Panel
        filter_group = QGroupBox("🔍 Data Filters")
        filter_layout = QVBoxLayout()

        filter_row = QHBoxLayout()

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

        self.active_filters_label = QLabel("No filters active")
        self.active_filters_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        self.active_filters_label.setWordWrap(True)
        filter_layout.addWidget(self.active_filters_label)

        # ========================
        # FASE 2: FILTRI RAPIDI
        # ========================
        quick_filter_row = QHBoxLayout()

        # Filtro temporale rapido
        quick_filter_row.addWidget(QLabel("Time Filter:"))
        self.quick_time_filter = QComboBox()
        self.quick_time_filter.addItems(list(self.TIME_FILTER_OPTIONS.keys()))
        self.quick_time_filter.setToolTip("Filtra i dati per finestra temporale rolling")
        self.quick_time_filter.currentTextChanged.connect(self._on_quick_filter_changed)
        self.quick_time_filter.setMaximumWidth(100)
        quick_filter_row.addWidget(self.quick_time_filter)

        # Limitatore Top/Worst N (per bar chart)
        self.top_worst_combo = QComboBox()
        self.top_worst_combo.addItems(['Top', 'Worst'])
        self.top_worst_combo.setToolTip("Top = valori più alti, Worst = valori più bassi")
        self.top_worst_combo.currentTextChanged.connect(self._on_quick_filter_changed)
        self.top_worst_combo.setMaximumWidth(70)
        quick_filter_row.addWidget(self.top_worst_combo)

        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(0, 500)
        self.top_n_spin.setValue(0)  # 0 = no limit
        self.top_n_spin.setSpecialValueText("All")
        self.top_n_spin.setToolTip("Limita categorie nel chart (0 = tutte)")
        self.top_n_spin.valueChanged.connect(self._on_quick_filter_changed)
        self.top_n_spin.setMaximumWidth(80)
        quick_filter_row.addWidget(self.top_n_spin)

        # Checkbox downsampling (Fase 3)
        self.downsample_check = QCheckBox("Auto Downsample")
        self.downsample_check.setChecked(True)
        self.downsample_check.setToolTip(f"Riduce automaticamente i punti a {self.MAX_POINTS_LINE_CHART} per line/scatter")
        quick_filter_row.addWidget(self.downsample_check)

        quick_filter_row.addStretch()
        filter_layout.addLayout(quick_filter_row)

        filter_group.setLayout(filter_layout)
        settings_layout.addWidget(filter_group)

        # Chart Configuration
        config_group = QGroupBox("📊 Chart Configuration")
        config_layout = QVBoxLayout()

        # Mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))

        self.static_radio = QRadioButton("Static Snapshot")
        self.static_radio.setChecked(True)
        self.static_radio.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.static_radio)

        self.time_evolution_radio = QRadioButton("Time Evolution")
        self.time_evolution_radio.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self.time_evolution_radio)

        mode_row.addStretch()
        config_layout.addLayout(mode_row)

        # Static mode controls
        self.static_controls = QWidget()
        static_layout = QVBoxLayout(self.static_controls)
        static_layout.setContentsMargins(0, 5, 0, 5)

        static_row1 = QHBoxLayout()
        static_row1.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(['bar', 'line', 'scatter'])
        static_row1.addWidget(self.chart_type_combo)

        static_row1.addWidget(QLabel("X:"))
        self.static_x_combo = QComboBox()
        self.static_x_combo.addItems(['None'])
        static_row1.addWidget(self.static_x_combo)

        static_row1.addWidget(QLabel("Y:"))
        self.static_y_combo = QComboBox()
        self.static_y_combo.addItems(['None'])
        static_row1.addWidget(self.static_y_combo)

        static_layout.addLayout(static_row1)

        static_row2 = QHBoxLayout()
        static_row2.addWidget(QLabel("Group By:"))
        self.static_group_combo = QComboBox()
        self.static_group_combo.addItems(['None'])
        static_row2.addWidget(self.static_group_combo)

        static_row2.addWidget(QLabel("Aggregation:"))
        self.static_agg_combo = QComboBox()
        self.static_agg_combo.addItems(['sum', 'mean', 'count', 'min', 'max', 'std', 'median'])
        static_row2.addWidget(self.static_agg_combo)

        static_row2.addStretch()
        static_layout.addLayout(static_row2)

        config_layout.addWidget(self.static_controls)

        # Time evolution controls
        self.time_controls = QWidget()
        time_layout = QVBoxLayout(self.time_controls)
        time_layout.setContentsMargins(0, 5, 0, 5)

        time_row1 = QHBoxLayout()
        time_row1.addWidget(QLabel("Time Column:"))
        self.time_column_combo = QComboBox()
        self.time_column_combo.addItems(['None'])
        time_row1.addWidget(self.time_column_combo)

        time_row1.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['None'])
        time_row1.addWidget(self.metric_combo)

        time_layout.addLayout(time_row1)

        time_row2 = QHBoxLayout()
        time_row2.addWidget(QLabel("Operation:"))
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(['cumsum', 'cumcount', 'rolling_mean', 'rolling_sum', 'raw'])
        self.operation_combo.currentTextChanged.connect(self._on_operation_changed)
        time_row2.addWidget(self.operation_combo)

        time_row2.addWidget(QLabel("Group By:"))
        self.time_group_combo = QComboBox()
        self.time_group_combo.addItems(['None'])
        time_row2.addWidget(self.time_group_combo)

        time_row2.addWidget(QLabel("Window:"))
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setRange(2, 1000)
        self.window_spinbox.setValue(20)
        self.window_spinbox.setMaximumWidth(80)
        self.window_spinbox.setEnabled(False)
        time_row2.addWidget(self.window_spinbox)

        time_row2.addStretch()
        time_layout.addLayout(time_row2)

        config_layout.addWidget(self.time_controls)
        self.time_controls.setVisible(False)

        # Buttons
        btn_row = QHBoxLayout()

        self.auto_update_checkbox = QCheckBox("Auto-update")
        self.auto_update_checkbox.setChecked(True)
        btn_row.addWidget(self.auto_update_checkbox)

        self.apply_btn = QPushButton("Apply Chart")
        self.apply_btn.clicked.connect(self.apply_chart)
        btn_row.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chart)
        btn_row.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export PNG")
        self.export_btn.clicked.connect(self._export_png)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()
        config_layout.addLayout(btn_row)

        config_group.setLayout(config_layout)
        settings_layout.addWidget(config_group)

        layout.addWidget(self.settings_container)

        # Chart Canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Info label
        self.info_label = QLabel("Configure chart and click 'Apply Chart'")
        self.info_label.setStyleSheet("font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)

    def _toggle_settings(self):
        """Mostra/nascondi pannello settings"""
        if self.settings_container.isVisible():
            self.settings_container.setVisible(False)
            self.toggle_settings_btn.setText("▶ Show Settings")
        else:
            self.settings_container.setVisible(True)
            self.toggle_settings_btn.setText("▼ Hide Settings")

    def _on_mode_changed(self):
        """Cambia visibilità controlli in base al mode"""
        if self.static_radio.isChecked():
            self.mode = 'static'
            self.static_controls.setVisible(True)
            self.time_controls.setVisible(False)
        else:
            self.mode = 'time_evolution'
            self.static_controls.setVisible(False)
            self.time_controls.setVisible(True)

    def _on_operation_changed(self, operation):
        """Abilita/disabilita window spinbox per rolling operations"""
        self.window_spinbox.setEnabled(operation.startswith('rolling'))

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
            if self.current_config and self.auto_update_checkbox.isChecked():
                self._apply_chart_from_config(self.current_config)

    def _clear_filters(self):
        """Pulisce tutti i filtri"""
        self.active_filter = None
        self._update_filter_label()
        if self.current_config and self.auto_update_checkbox.isChecked():
            self._apply_chart_from_config(self.current_config)

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

            if len(filter_str) > 80:
                filter_str = filter_str[:77] + "..."

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
        """Applica filtri al DataFrame"""
        if not self.active_filter or not self.active_filter.conditions:
            return df

        mask = self.active_filter.apply(df)
        return df[mask]

    def set_data(self, df: pd.DataFrame):
        """
        Imposta i dati sorgente con debounce (Fase 1).

        NON triggera rendering immediato. Avvia invece un timer che,
        se non interrotto da nuovi dati, eseguirà il rendering.

        Preprocessing datetime eseguito qui (una sola volta).
        """
        if df is None:
            df = pd.DataFrame()

        # ========================
        # FASE 1: PREPROCESSING DATETIME
        # ========================
        if not df.empty:
            df = self._preprocess_datetime(df)

        self.source_data = df

        if df.empty:
            return

        # Salva selezioni correnti (o da pending config)
        if self._pending_config:
            current_static_x = self._pending_config.get('static_x', 'None')
            current_static_y = self._pending_config.get('static_y', 'None')
            current_static_group = self._pending_config.get('static_group', 'None')
            current_time_col = self._pending_config.get('time_column', 'None')
            current_metric = self._pending_config.get('metric', 'None')
            current_time_group = self._pending_config.get('time_group', 'None')
        else:
            current_static_x = self.static_x_combo.currentText()
            current_static_y = self.static_y_combo.currentText()
            current_static_group = self.static_group_combo.currentText()
            current_time_col = self.time_column_combo.currentText()
            current_metric = self.metric_combo.currentText()
            current_time_group = self.time_group_combo.currentText()

        # Aggiorna combo boxes
        columns = ['None'] + list(df.columns)
        numeric_cols = ['None'] + list(df.select_dtypes(include=['number']).columns)

        self.static_x_combo.clear()
        self.static_x_combo.addItems(columns)

        self.static_y_combo.clear()
        self.static_y_combo.addItems(numeric_cols)

        self.static_group_combo.clear()
        self.static_group_combo.addItems(columns)

        self.time_column_combo.clear()
        self.time_column_combo.addItems(columns)

        self.metric_combo.clear()
        self.metric_combo.addItems(numeric_cols)

        self.time_group_combo.clear()
        self.time_group_combo.addItems(columns)

        # Ripristina selezioni
        if current_static_x in columns:
            self.static_x_combo.setCurrentText(current_static_x)
        if current_static_y in numeric_cols:
            self.static_y_combo.setCurrentText(current_static_y)
        if current_static_group in columns:
            self.static_group_combo.setCurrentText(current_static_group)
        if current_time_col in columns:
            self.time_column_combo.setCurrentText(current_time_col)
        if current_metric in numeric_cols:
            self.metric_combo.setCurrentText(current_metric)
        if current_time_group in columns:
            self.time_group_combo.setCurrentText(current_time_group)

        # Applica pending config se presente
        if self._pending_config:
            self._apply_pending_config()
            self._pending_config = None
        # ========================
        # FASE 1: DEBOUNCE AUTO-UPDATE
        # ========================
        # Invece di chiamare direttamente _apply_chart_from_config,
        # avviamo il timer di debounce
        elif self.current_config and self.auto_update_checkbox.isChecked():
            self._debounce_timer.stop()
            self._debounce_timer.start(self.DEBOUNCE_MS)

    def _apply_pending_config(self):
        """Applica la configurazione pending dopo che le combo box sono state popolate."""
        config = self._pending_config
        if not config:
            return

        # Mode
        if config.get('static_mode', True):
            self.static_radio.setChecked(True)
            self.mode = 'static'
        elif config.get('time_evolution_mode', False):
            self.time_evolution_radio.setChecked(True)
            self.mode = 'time_evolution'

        # Chart type
        if 'chart_type' in config:
            index = self.chart_type_combo.findText(config['chart_type'])
            if index >= 0:
                self.chart_type_combo.setCurrentIndex(index)

        # Static controls
        if 'static_x' in config and config['static_x']:
            index = self.static_x_combo.findText(config['static_x'])
            if index >= 0:
                self.static_x_combo.setCurrentIndex(index)

        if 'static_y' in config and config['static_y']:
            index = self.static_y_combo.findText(config['static_y'])
            if index >= 0:
                self.static_y_combo.setCurrentIndex(index)

        if 'static_group' in config and config['static_group']:
            index = self.static_group_combo.findText(config['static_group'])
            if index >= 0:
                self.static_group_combo.setCurrentIndex(index)

        if 'static_agg' in config:
            index = self.static_agg_combo.findText(config['static_agg'])
            if index >= 0:
                self.static_agg_combo.setCurrentIndex(index)

        # Time evolution controls
        if 'time_column' in config and config['time_column']:
            index = self.time_column_combo.findText(config['time_column'])
            if index >= 0:
                self.time_column_combo.setCurrentIndex(index)

        if 'metric' in config and config['metric']:
            index = self.metric_combo.findText(config['metric'])
            if index >= 0:
                self.metric_combo.setCurrentIndex(index)

        if 'operation' in config:
            index = self.operation_combo.findText(config['operation'])
            if index >= 0:
                self.operation_combo.setCurrentIndex(index)

        if 'time_group' in config and config['time_group']:
            index = self.time_group_combo.findText(config['time_group'])
            if index >= 0:
                self.time_group_combo.setCurrentIndex(index)

        if 'window' in config:
            self.window_spinbox.setValue(config['window'])

        if 'auto_update' in config:
            self.auto_update_checkbox.setChecked(config['auto_update'])

        # Quick filters (Fase 2-3)
        if 'quick_time_filter' in config:
            index = self.quick_time_filter.findText(config['quick_time_filter'])
            if index >= 0:
                self.quick_time_filter.setCurrentIndex(index)

        if 'top_worst' in config:
            index = self.top_worst_combo.findText(config['top_worst'])
            if index >= 0:
                self.top_worst_combo.setCurrentIndex(index)

        if 'top_n' in config:
            self.top_n_spin.setValue(config['top_n'])

        if 'downsample' in config:
            self.downsample_check.setChecked(config['downsample'])

        # Applica current_config per il chart
        if config.get('current_config'):
            self.current_config = config['current_config']
            self._apply_chart_from_config(self.current_config)

    # ========================
    # FASE 1: METODI THROTTLING
    # ========================

    def _preprocess_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte colonne datetime una sola volta.
        Risparmia tempo nei metodi di plot.
        """
        df = df.copy()

        # Cerca colonne datetime comuni
        datetime_cols = ['timestamp', 'time', 'datetime', 'date', 'created_at', 'updated_at']

        for col in datetime_cols:
            if col in df.columns:
                if df[col].dtype == 'object' or str(df[col].dtype) == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        self.logger.debug(f"Could not convert {col} to datetime: {e}")

        return df

    def _on_debounce_timeout(self):
        """
        Chiamato quando il timer di debounce scade.
        Esegue il rendering se non già in corso.
        """
        if self._is_rendering:
            # Rendering in corso, segna come pending
            self._pending_render = True
            return

        self._execute_render()

    def _execute_render(self):
        """
        Esegue il rendering protetto dal flag _is_rendering.
        """
        if self.source_data.empty or not self.current_config:
            return

        # Imposta flag rendering
        self._is_rendering = True
        self._pending_render = False

        try:
            self._apply_chart_from_config(self.current_config)
        except Exception as e:
            self.logger.error(f"Error during render: {e}")
            self.info_label.setText(f"Render error: {str(e)[:50]}")
        finally:
            self._is_rendering = False

            # Se c'era un render pending, eseguilo dopo un breve delay
            if self._pending_render:
                QTimer.singleShot(50, self._execute_render)

    def _on_quick_filter_changed(self):
        """Callback quando cambiano i filtri rapidi (Fase 2)."""
        if self.auto_update_checkbox.isChecked() and self.current_config:
            self._debounce_timer.stop()
            self._debounce_timer.start(self.DEBOUNCE_MS)

    # ========================
    # FASE 2: FILTRI ROLLING E TOP N
    # ========================

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
        Applica filtro temporale rolling (Fase 2).
        Riduce i dati "alla fonte" per alleggerire Matplotlib.
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

    def _apply_top_n_limit(self, df: pd.DataFrame, x_col: str, y_col: str, agg: str) -> pd.DataFrame:
        """
        Applica limitatore Top/Worst N per bar chart (Fase 2).
        Dopo groupby, ordina e prende solo i primi/ultimi N.
        """
        top_n = self.top_n_spin.value()
        is_worst = self.top_worst_combo.currentText() == 'Worst'

        if top_n == 0 or df.empty:
            return df

        # Raggruppa e aggrega
        if x_col in df.columns and y_col in df.columns:
            grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
            grouped.columns = [x_col, y_col]

            # Ordina per valore Y (ascending per Worst, descending per Top)
            grouped = grouped.sort_values(by=y_col, ascending=is_worst).head(top_n)

            return grouped

        return df

    # ========================
    # FASE 3: DOWNSAMPLING
    # ========================

    def _downsample_data(self, df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """
        Downsampling intelligente per line chart e scatter (Fase 3).
        Mantiene la forma del trend riducendo i punti.
        """
        if not self.downsample_check.isChecked():
            return df

        if len(df) <= self.MAX_POINTS_LINE_CHART:
            return df

        # Calcola fattore di campionamento
        factor = len(df) // self.MAX_POINTS_LINE_CHART

        if factor <= 1:
            return df

        # Campiona ogni N righe
        result = df.iloc[::factor].copy()

        self.logger.debug(f"Downsampled from {len(df)} to {len(result)} points (factor: {factor})")

        return result

    def apply_chart(self):
        """Applica la configurazione chart"""
        if self.source_data.empty:
            QMessageBox.warning(self, "No Data", "No source data available")
            return

        if self.mode == 'static':
            config = self._get_static_config()
        else:
            config = self._get_time_evolution_config()

        if not config:
            return

        self.current_config = config
        self._apply_chart_from_config(config)

    def _get_static_config(self) -> Optional[Dict[str, Any]]:
        """Ottieni configurazione per static mode"""
        x_col = self.static_x_combo.currentText()
        y_col = self.static_y_combo.currentText()
        group_by = self.static_group_combo.currentText()
        chart_type = self.chart_type_combo.currentText()
        agg = self.static_agg_combo.currentText()

        if x_col == 'None' or y_col == 'None':
            QMessageBox.warning(self, "Invalid Config", "Please select X and Y columns")
            return None

        return {
            'mode': 'static',
            'chart_type': chart_type,
            'x_column': x_col,
            'y_column': y_col,
            'group_by': group_by if group_by != 'None' else None,
            'aggregation': agg
        }

    def _get_time_evolution_config(self) -> Optional[Dict[str, Any]]:
        """Ottieni configurazione per time evolution mode"""
        time_col = self.time_column_combo.currentText()
        metric = self.metric_combo.currentText()
        operation = self.operation_combo.currentText()
        group_by = self.time_group_combo.currentText()
        window = self.window_spinbox.value()

        if time_col == 'None' or metric == 'None':
            QMessageBox.warning(self, "Invalid Config", "Please select Time Column and Metric")
            return None

        return {
            'mode': 'time_evolution',
            'time_column': time_col,
            'metric_column': metric,
            'operation': operation,
            'group_by': group_by if group_by != 'None' else None,
            'window': window
        }

    def _apply_chart_from_config(self, config: Dict[str, Any]):
        """Applica chart usando configurazione salvata con ottimizzazioni."""
        try:
            # ========================
            # FASE 2: APPLICA FILTRI IN CASCATA
            # ========================
            # 1. Filtri avanzati (AND/OR)
            filtered_data = self._apply_filters_to_data(self.source_data)

            # 2. Filtro temporale rolling
            filtered_data = self._apply_time_filter(filtered_data)

            if filtered_data.empty:
                # Non mostrare warning popup per evitare flood
                self.info_label.setText("No data after filtering")
                self.figure.clear()
                self.canvas.draw()
                return

            # Clear figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Salva info per il label
            original_count = len(self.source_data)
            filtered_count = len(filtered_data)

            if config['mode'] == 'static':
                self._plot_static(ax, filtered_data, config)
            else:
                self._plot_time_evolution(ax, filtered_data, config)

            self.canvas.draw()
            self._last_chart_mode = config['mode']

            # Update info con dettagli filtri
            info_parts = [f"Mode: {config['mode']}"]

            if filtered_count < original_count:
                info_parts.append(f"Data: {filtered_count}/{original_count}")

            time_filter = self.quick_time_filter.currentText()
            if time_filter != 'All':
                info_parts.append(f"Time: {time_filter}")

            top_n = self.top_n_spin.value()
            if top_n > 0:
                top_worst = self.top_worst_combo.currentText()
                info_parts.append(f"{top_worst} {top_n}")

            self.info_label.setText(" | ".join(info_parts))

        except Exception as e:
            self.logger.error(f"Chart error: {e}")
            self.info_label.setText(f"Chart error: {str(e)[:60]}")
            import traceback
            traceback.print_exc()

    def _plot_static(self, ax, df: pd.DataFrame, config: Dict[str, Any]):
        """Plot static snapshot con Top N e downsampling (Fase 2 e 3)."""
        x_col = config['x_column']
        y_col = config['y_column']
        group_by = config['group_by']
        agg = config['aggregation']
        chart_type = config['chart_type']

        top_n = self.top_n_spin.value()
        is_worst = self.top_worst_combo.currentText() == 'Worst'

        if group_by:
            pivot = df.pivot_table(
                values=y_col,
                index=x_col,
                columns=group_by,
                aggfunc=agg,
                fill_value=0
            )

            # ========================
            # FASE 2: APPLICA TOP/WORST N AL PIVOT
            # ========================
            if top_n > 0 and len(pivot) > top_n:
                # Ordina per somma totale delle righe
                row_sums = pivot.sum(axis=1)
                if is_worst:
                    top_indices = row_sums.nsmallest(top_n).index
                else:
                    top_indices = row_sums.nlargest(top_n).index
                pivot = pivot.loc[top_indices]

            if chart_type == 'bar':
                pivot.plot(kind='bar', ax=ax, width=0.7)
            elif chart_type == 'line':
                # Fase 3: downsampling per line
                if len(pivot) > self.MAX_POINTS_LINE_CHART and self.downsample_check.isChecked():
                    factor = len(pivot) // self.MAX_POINTS_LINE_CHART
                    pivot = pivot.iloc[::max(factor, 1)]
                pivot.plot(kind='line', ax=ax, marker='o', markersize=3)
            elif chart_type == 'scatter':
                # Fase 3: downsampling per scatter
                if len(pivot) > self.MAX_POINTS_LINE_CHART and self.downsample_check.isChecked():
                    factor = len(pivot) // self.MAX_POINTS_LINE_CHART
                    pivot = pivot.iloc[::max(factor, 1)]
                for col in pivot.columns:
                    ax.scatter(pivot.index, pivot[col], label=col, s=30, alpha=0.6)
        else:
            grouped = df.groupby(x_col)[y_col].agg(agg)

            # ========================
            # FASE 2: APPLICA TOP/WORST N AL GROUPED
            # ========================
            if top_n > 0 and len(grouped) > top_n:
                if is_worst:
                    grouped = grouped.nsmallest(top_n)
                else:
                    grouped = grouped.nlargest(top_n)

            if chart_type == 'bar':
                grouped.plot(kind='bar', ax=ax, width=0.7)
            elif chart_type == 'line':
                # Fase 3: downsampling per line
                if len(grouped) > self.MAX_POINTS_LINE_CHART and self.downsample_check.isChecked():
                    factor = len(grouped) // self.MAX_POINTS_LINE_CHART
                    grouped = grouped.iloc[::max(factor, 1)]
                grouped.plot(kind='line', ax=ax, marker='o', markersize=3)
            elif chart_type == 'scatter':
                # Fase 3: downsampling per scatter
                if len(grouped) > self.MAX_POINTS_LINE_CHART and self.downsample_check.isChecked():
                    factor = len(grouped) // self.MAX_POINTS_LINE_CHART
                    grouped = grouped.iloc[::max(factor, 1)]
                ax.scatter(grouped.index, grouped.values, s=30, alpha=0.6)

        ax.set_xlabel(x_col)
        ax.set_ylabel(f"{agg}({y_col})")
        ax.set_title(f"{agg.capitalize()} of {y_col} by {x_col}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotazione etichette se troppe
        if len(ax.get_xticklabels()) > 10:
            ax.tick_params(axis='x', rotation=45)

        # Gestione safe di tight_layout
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception:
            # Se tight_layout fallisce, usa subplots_adjust come fallback
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    def _plot_time_evolution(self, ax, df: pd.DataFrame, config: Dict[str, Any]):
        """Plot time evolution con downsampling (Fase 3)."""
        time_col = config['time_column']
        metric_col = config['metric_column']
        operation = config['operation']
        group_by = config['group_by']
        window = config['window']

        # Converti timestamp (già preprocessato in set_data, ma verifica)
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            pass

        df = df.sort_values(time_col).copy()

        # ========================
        # FASE 3: DOWNSAMPLING INTELLIGENTE
        # ========================
        original_len = len(df)
        if len(df) > self.MAX_POINTS_LINE_CHART and self.downsample_check.isChecked():
            factor = len(df) // self.MAX_POINTS_LINE_CHART
            df = df.iloc[::max(factor, 1)].copy()
            self.logger.debug(f"Time evolution downsampled: {original_len} -> {len(df)} points")

        if group_by:
            all_times = df[time_col].unique()
            all_times = pd.Series(all_times).sort_values().values

            # Limita il numero di gruppi se troppi
            unique_groups = df[group_by].unique()
            top_n = self.top_n_spin.value()
            if top_n > 0 and len(unique_groups) > top_n:
                # Prendi i gruppi con più dati
                group_counts = df[group_by].value_counts().head(top_n)
                unique_groups = group_counts.index.tolist()

            for group_value in unique_groups:
                subset = df[df[group_by] == group_value].copy()
                full_series = pd.Series(index=all_times, dtype=float)

                subset_sorted = subset.sort_values(time_col).reset_index(drop=True)
                y_cumsum = self._apply_operation(subset_sorted[metric_col], operation, window)

                for i, (ts, val) in enumerate(zip(subset_sorted[time_col], y_cumsum)):
                    full_series[ts] = val

                full_series = full_series.ffill()
                line, = ax.plot(full_series.index, full_series.values, label=str(group_value), linewidth=1.5, marker='.', markersize=2)
                self._line_objects[str(group_value)] = line
        else:
            y_values = self._apply_operation(df[metric_col], operation, window)
            line, = ax.plot(df[time_col], y_values, linewidth=1.5, marker='.', markersize=2)
            self._line_objects['main'] = line

        op_label = operation.replace('_', ' ').title()
        if operation.startswith('rolling'):
            op_label += f" (w={window})"

        ax.set_xlabel(time_col)
        ax.set_ylabel(f"{op_label} of {metric_col}")
        ax.set_title(f"{op_label} - {metric_col} over Time")

        if group_by:
            ax.legend(loc='upper left', fontsize='small')

        ax.grid(True, alpha=0.3)

        try:
            import matplotlib.dates as mdates
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                sample_time = df[time_col].iloc[0]
                if hasattr(sample_time, 'date'):
                    unique_dates = df[time_col].dt.date.nunique() if hasattr(df[time_col].dt, 'date') else 1

                    if unique_dates == 1:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    else:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            self.figure.autofmt_xdate(rotation=45)
        except Exception:
            ax.tick_params(axis='x', rotation=45)

        # Gestione safe di tight_layout
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception:
            # Se tight_layout fallisce, usa subplots_adjust come fallback
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    def _apply_operation(self, series: pd.Series, operation: str, window: int) -> pd.Series:
        """Applica operation alla serie"""
        if operation == 'cumsum':
            return series.cumsum()
        elif operation == 'cumcount':
            return pd.Series(range(1, len(series) + 1), index=series.index)
        elif operation == 'rolling_mean':
            return series.rolling(window=window, min_periods=1).mean()
        elif operation == 'rolling_sum':
            return series.rolling(window=window, min_periods=1).sum()
        elif operation == 'raw':
            return series
        else:
            return series

    def clear_chart(self):
        """Pulisce il chart e resetta lo stato interno."""
        self.current_config = None
        self._line_objects.clear()  # Reset riferimenti linee (Fase 3)
        self._last_chart_mode = None
        self.figure.clear()
        self.canvas.draw()
        self.info_label.setText("Chart cleared")

    def _export_png(self):
        """Esporta chart come PNG"""
        if not self.current_config:
            QMessageBox.warning(self, "Export", "No chart to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chart",
            f"chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png)"
        )

        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export", f"Chart exported to {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    def get_config(self) -> dict:
        """Salva configurazione corrente (include nuove impostazioni Fase 2-3)."""
        return {
            'mode': self.mode,
            'static_mode': self.static_radio.isChecked(),
            'time_evolution_mode': self.time_evolution_radio.isChecked(),
            'chart_type': self.chart_type_combo.currentText(),
            'static_x': self.static_x_combo.currentText(),
            'static_y': self.static_y_combo.currentText(),
            'static_group': self.static_group_combo.currentText(),
            'static_agg': self.static_agg_combo.currentText(),
            'time_column': self.time_column_combo.currentText(),
            'metric': self.metric_combo.currentText(),
            'operation': self.operation_combo.currentText(),
            'time_group': self.time_group_combo.currentText(),
            'window': self.window_spinbox.value(),
            'auto_update': self.auto_update_checkbox.isChecked(),
            'settings_visible': hasattr(self, 'settings_container') and self.settings_container.isVisible(),
            'current_config': self.current_config.copy() if self.current_config else None,
            # Nuove impostazioni Fase 2-3
            'quick_time_filter': self.quick_time_filter.currentText(),
            'top_worst': self.top_worst_combo.currentText(),
            'top_n': self.top_n_spin.value(),
            'downsample': self.downsample_check.isChecked(),
        }

    def restore_config(self, config: dict):
        """
        Ripristina configurazione salvata.

        Se i dati non sono ancora disponibili (combo box vuote),
        salva la config come pending e la applica in set_data.
        """
        try:
            # Ripristina settings visibility subito (non dipende dai dati)
            if 'settings_visible' in config and hasattr(self, 'settings_container'):
                if config['settings_visible']:
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
            if config.get('static_mode', True):
                self.static_radio.setChecked(True)
                self.mode = 'static'
            elif config.get('time_evolution_mode', False):
                self.time_evolution_radio.setChecked(True)
                self.mode = 'time_evolution'

            if 'chart_type' in config:
                index = self.chart_type_combo.findText(config['chart_type'])
                if index >= 0:
                    self.chart_type_combo.setCurrentIndex(index)

            if 'static_x' in config and config['static_x']:
                index = self.static_x_combo.findText(config['static_x'])
                if index >= 0:
                    self.static_x_combo.setCurrentIndex(index)

            if 'static_y' in config and config['static_y']:
                index = self.static_y_combo.findText(config['static_y'])
                if index >= 0:
                    self.static_y_combo.setCurrentIndex(index)

            if 'static_group' in config and config['static_group']:
                index = self.static_group_combo.findText(config['static_group'])
                if index >= 0:
                    self.static_group_combo.setCurrentIndex(index)

            if 'static_agg' in config:
                index = self.static_agg_combo.findText(config['static_agg'])
                if index >= 0:
                    self.static_agg_combo.setCurrentIndex(index)

            if 'time_column' in config and config['time_column']:
                index = self.time_column_combo.findText(config['time_column'])
                if index >= 0:
                    self.time_column_combo.setCurrentIndex(index)

            if 'metric' in config and config['metric']:
                index = self.metric_combo.findText(config['metric'])
                if index >= 0:
                    self.metric_combo.setCurrentIndex(index)

            if 'operation' in config:
                index = self.operation_combo.findText(config['operation'])
                if index >= 0:
                    self.operation_combo.setCurrentIndex(index)

            if 'time_group' in config and config['time_group']:
                index = self.time_group_combo.findText(config['time_group'])
                if index >= 0:
                    self.time_group_combo.setCurrentIndex(index)

            if 'window' in config:
                self.window_spinbox.setValue(config['window'])

            if 'auto_update' in config:
                self.auto_update_checkbox.setChecked(config['auto_update'])

            # Quick filters (Fase 2-3)
            if 'quick_time_filter' in config:
                index = self.quick_time_filter.findText(config['quick_time_filter'])
                if index >= 0:
                    self.quick_time_filter.setCurrentIndex(index)

            if 'top_worst' in config:
                index = self.top_worst_combo.findText(config['top_worst'])
                if index >= 0:
                    self.top_worst_combo.setCurrentIndex(index)

            if 'top_n' in config:
                self.top_n_spin.setValue(config['top_n'])

            if 'downsample' in config:
                self.downsample_check.setChecked(config['downsample'])

            if config.get('current_config'):
                self.current_config = config['current_config']
                self._apply_chart_from_config(self.current_config)

        except Exception as e:
            self.logger.error(f"Error restoring chart config: {e}")
            import traceback
            traceback.print_exc()