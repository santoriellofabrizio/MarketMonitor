"""
ChartWidget con filtri avanzati AND/OR
- Mode: Static Snapshot | Time Evolution
- Filtri avanzati con logica AND/OR
- Auto-update
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QPushButton, QRadioButton,
                             QSpinBox, QCheckBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd
from typing import Optional, Dict, Any
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import sistema filtri avanzati
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.filter import AdvancedFilterDialog, FilterGroup


class ChartWidget(QWidget):
    """Widget versatile per chart con filtri avanzati e dual mode"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.source_data = pd.DataFrame()
        self.current_config = None
        self.mode = 'static'

        # Filtri avanzati
        self.active_filter: Optional[FilterGroup] = None

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
        """Imposta i dati sorgente"""
        self.source_data = df

        if df.empty:
            return

        # Salva selezioni correnti
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

        # Auto-update
        if self.current_config and self.auto_update_checkbox.isChecked():
            self._apply_chart_from_config(self.current_config)

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
        """Applica chart usando configurazione salvata"""
        try:
            # Applica filtri
            filtered_data = self._apply_filters_to_data(self.source_data)

            if filtered_data.empty:
                QMessageBox.warning(self, "No Data", "Filters resulted in empty dataset")
                self.clear_chart()
                return

            # Clear figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if config['mode'] == 'static':
                self._plot_static(ax, filtered_data, config)
            else:
                self._plot_time_evolution(ax, filtered_data, config)

            self.canvas.draw()

            # Update info
            filter_text = f" | Filtered: {len(filtered_data)} rows" if self.active_filter and self.active_filter.conditions else ""
            self.info_label.setText(f"Chart applied | Mode: {config['mode']}{filter_text}")

        except Exception as e:
            QMessageBox.warning(self, "Chart Error", f"Error creating chart: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_static(self, ax, df: pd.DataFrame, config: Dict[str, Any]):
        """Plot static snapshot"""
        x_col = config['x_column']
        y_col = config['y_column']
        group_by = config['group_by']
        agg = config['aggregation']
        chart_type = config['chart_type']

        if group_by:
            pivot = df.pivot_table(
                values=y_col,
                index=x_col,
                columns=group_by,
                aggfunc=agg,
                fill_value=0
            )

            if chart_type == 'bar':
                pivot.plot(kind='bar', ax=ax, width=0.7)
            elif chart_type == 'line':
                pivot.plot(kind='line', ax=ax, marker='o')
            elif chart_type == 'scatter':
                for col in pivot.columns:
                    ax.scatter(pivot.index, pivot[col], label=col, s=50, alpha=0.6)
        else:
            grouped = df.groupby(x_col)[y_col].agg(agg)

            if chart_type == 'bar':
                grouped.plot(kind='bar', ax=ax, width=0.7)
            elif chart_type == 'line':
                grouped.plot(kind='line', ax=ax, marker='o')
            elif chart_type == 'scatter':
                ax.scatter(grouped.index, grouped.values, s=50, alpha=0.6)

        ax.set_xlabel(x_col)
        ax.set_ylabel(f"{agg}({y_col})")
        ax.set_title(f"{agg.capitalize()} of {y_col} by {x_col}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gestione safe di tight_layout
        try:
            self.figure.tight_layout(pad=1.5)
        except Exception:
            # Se tight_layout fallisce, usa subplots_adjust come fallback
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    def _plot_time_evolution(self, ax, df: pd.DataFrame, config: Dict[str, Any]):
        """Plot time evolution"""
        time_col = config['time_column']
        metric_col = config['metric_column']
        operation = config['operation']
        group_by = config['group_by']
        window = config['window']

        # Converti timestamp
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
        except:
            pass

        df = df.sort_values(time_col).copy()

        if group_by:
            all_times = df[time_col].unique()
            all_times = pd.Series(all_times).sort_values().values

            for group_value in df[group_by].unique():
                subset = df[df[group_by] == group_value].copy()
                full_series = pd.Series(index=all_times, dtype=float)

                subset_sorted = subset.sort_values(time_col).reset_index(drop=True)
                y_cumsum = self._apply_operation(subset_sorted[metric_col], operation, window)

                for i, (ts, val) in enumerate(zip(subset_sorted[time_col], y_cumsum)):
                    full_series[ts] = val

                full_series = full_series.ffill()
                ax.plot(full_series.index, full_series.values, label=str(group_value), linewidth=2)
        else:
            y_values = self._apply_operation(df[metric_col], operation, window)
            ax.plot(df[time_col], y_values, linewidth=2)

        op_label = operation.replace('_', ' ').title()
        if operation.startswith('rolling'):
            op_label += f" (w={window})"

        ax.set_xlabel(time_col)
        ax.set_ylabel(f"{op_label} of {metric_col}")
        ax.set_title(f"{op_label} - {metric_col} over Time")

        if group_by:
            ax.legend()

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
        """Pulisce il chart"""
        self.current_config = None
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
        """Salva configurazione corrente"""
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
        }

    def restore_config(self, config: dict):
        """Ripristina configurazione salvata"""
        try:
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

            if 'settings_visible' in config and hasattr(self, 'settings_container'):
                if config['settings_visible']:
                    self.settings_container.show()
                    self.toggle_settings_btn.setText("▼ Hide Settings")
                else:
                    self.settings_container.hide()
                    self.toggle_settings_btn.setText("▶ Show Settings")

            if config.get('current_config'):
                self.current_config = config['current_config']
                if not self.source_data.empty:
                    self._apply_chart_from_config(self.current_config)

        except Exception as e:
            print(f"Error restoring chart config: {e}")
            import traceback
            traceback.print_exc()