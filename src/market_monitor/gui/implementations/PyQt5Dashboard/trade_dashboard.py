"""
Dashboard completa per trades (PyQt5Dashboard) - CON WORKER THREAD

Dashboard PyQt5 per la visualizzazione real-time dei trade con supporto
a worker thread in background.

Supporta due modalità di alimentazione dati:
- queue  : polling da Queue interna allo stesso processo Strategy
- redis  : sottoscrizione RedisPublisher Pub/Sub da processo separato

Caratteristiche:
- Tabella Raw Trades con deduplicazione e filtri
- Metriche aggregate globali (calcolate su ALL data)
- Finestre detachable per Pivot, Chart e Flow Monitor
- Separazione rigorosa tra worker thread e UI thread
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QDialog, QListWidget,
    QDialogButtonBox, QListWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
import pandas as pd
from typing import Optional, Dict, Union
import logging

from market_monitor.gui.implementations.PyQt5Dashboard.base import BasePyQt5Dashboard
from market_monitor.gui.implementations.PyQt5Dashboard.common import safe_concat
from market_monitor.gui.implementations.PyQt5Dashboard.dashboard_extension import TradeDashboardExtensions
from market_monitor.gui.implementations.PyQt5Dashboard.detached_windows import (
    DetachedPivotWindow,
    DetachedChartWindow,
    DetachedFlowWindow
)
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.dashboard_state import DashboardState
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.trade_table import TradeTableWidget
from market_monitor.gui.implementations.PyQt5Dashboard.metrics_definition import METRIC_DEFINITIONS
from market_monitor.gui.threaded_GUI.QueueDataSource import QueueDataSource
from market_monitor.gui.implementations.PyQt5Dashboard.worker_thread import (
    QueuePollingThread,
    RedisPubSubThread
)


class TradeDashboard(BasePyQt5Dashboard, TradeDashboardExtensions):
    """
    Dashboard principale per la visualizzazione dei trade.

    Responsabilità:
    - Gestione lifecycle UI
    - Avvio/arresto worker thread
    - Accumulo e deduplicazione dei trade
    - Aggiornamento metriche globali
    - Coordinamento finestre detachable

    Tutti gli aggiornamenti UI avvengono nel MAIN THREAD tramite
    signal/slot Qt. Il worker thread non interagisce mai direttamente
    con widget grafici.
    """

    def __init__(
            self,
            datasource: Optional[QueueDataSource] = None,
            mode: str = "redis",
            redis_config: Optional[Dict] = None,
            columns: Optional[list] = None,
            logger: Optional[logging.Logger] = None,
            config: Optional[Dict] = None,
            dedup_column: str = "trade_index",
            datetime_columns: Union[list, str] = "timestamp",
            datetime_format: str = "%H:%M:%S.%f",
            metrics_config: Optional[Dict] = None,
    ):
        self.config = config or {}
        self.mode = mode
        self.redis_config = redis_config or {}
        self.columns = columns

        self.all_trades = pd.DataFrame()
        self.current_filtered_data = pd.DataFrame()

        self.dashboard_state = DashboardState()
        self.worker_thread = None

        self.dedup_column = dedup_column
        self.datetime_columns = datetime_columns
        self.datetime_format = datetime_format

        # ===== METRICS CONFIG =====
        self.metrics_config = metrics_config or {}
        self.metrics_enabled = self.metrics_config.get("enabled", True)
        self.metric_items = self.metrics_config.get("items", [])
        
        # Default metrics se lista vuota
        if self.metrics_enabled and not self.metric_items:
            self.metric_items = ["total_trades", "own_trades", "spread_pl_sum"]
        
        self.metric_labels = {}

        # ===== DETACHED WINDOWS =====
        self.detached_pivots = []
        self.detached_charts = []
        self.detached_flows = []

        super().__init__(
            datasource=datasource,
            title="Trade Dashboard",
            logger=logger
        )

        # Setup menu Dashboard
        self._setup_dashboard_menu()

    def start(self):
        """
        Avvia la dashboard mostrando la finestra e iniziando
        il monitoraggio dei dati.
        """
        self.logger.info("Starting TradeDashboard")
        self.show()
        self.start_monitoring()
        return self

    def setup_ui(self):
        # Controls panel
        controls = self._create_controls()
        self.main_layout.addWidget(controls)

        # Metrics panel
        if self.metrics_enabled:
            self.metrics_panel = self._create_metrics_panel()
            self.main_layout.addWidget(self.metrics_panel)

        # Trade table
        self.trade_table = TradeTableWidget(
            columns=self.columns,
            dedup_column=self.dedup_column,
            datetime_columns=self.datetime_columns,
            datetime_format=self.datetime_format
        )
        self.trade_table.filtered_data_changed.connect(
            self._on_filtered_data_changed
        )

        self.main_layout.addWidget(self.trade_table)

    def start_monitoring(self):
        """
        Crea e avvia il worker thread per la ricezione dei dati.
        """
        if self.worker_thread is not None:
            self.logger.warning("Monitoring already started")
            return

        self.logger.info(f"Starting monitoring in {self.mode} mode")

        if self.mode == 'queue':
            if self.datasource is None:
                raise ValueError("Queue mode requires datasource")
            self.worker_thread = QueuePollingThread(self.datasource)

        elif self.mode == 'redis':
            self.worker_thread = RedisPubSubThread(self.redis_config)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.worker_thread.data_updated.connect(self._on_data_received)
        self.worker_thread.status_updated.connect(self._on_status_updated)
        self.worker_thread.error_occurred.connect(self._on_error)

        self.worker_thread.start()

    def _create_controls(self) -> QWidget:
        """Crea il pannello dei controlli."""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(5, 5, 5, 5)

        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        layout.addWidget(self.pause_btn)

        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.clear_all)
        layout.addWidget(clear_btn)

        columns_btn = QPushButton("📋 Choose Columns")
        columns_btn.clicked.connect(self._show_column_chooser)
        layout.addWidget(columns_btn)

        detach_pivot_btn = QPushButton("🪟 New Pivot Window")
        detach_pivot_btn.clicked.connect(self._create_detached_pivot)
        layout.addWidget(detach_pivot_btn)

        detach_chart_btn = QPushButton("🪟 New Chart Window")
        detach_chart_btn.clicked.connect(self._create_detached_chart)
        layout.addWidget(detach_chart_btn)

        detach_flow_btn = QPushButton("🪟 New Flow Monitor")
        detach_flow_btn.clicked.connect(self._create_detached_flow)
        layout.addWidget(detach_flow_btn)

        layout.addStretch()
        return controls

    def _toggle_pause(self):
        """Gestisce pausa/ripresa del flusso dati."""
        if self.paused:
            self.resume()
            self.pause_btn.setText("⏸️ Pause")
        else:
            self.pause()
            self.pause_btn.setText("▶️ Resume")

    def _on_data_received(self, df: pd.DataFrame):
        """
        Ricezione nuovi dati dal worker thread.
        Esegue SEMPRE nel main thread Qt.
        """
        if self.paused:
            return

        self.logger.debug(f"Received {len(df)} new trades")

        # NORMALIZZAZIONE TIMESTAMP: Assicurati che timestamp sia SEMPRE datetime pandas
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype != 'datetime64[ns]':
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        if self.all_trades.empty:
            self.all_trades = df.copy()
        else:
            # NORMALIZZA anche all_trades se necessario
            if 'timestamp' in self.all_trades.columns:
                if self.all_trades['timestamp'].dtype != 'datetime64[ns]':
                    self.logger.info("Normalizing all_trades timestamps to datetime")
                    self.all_trades['timestamp'] = pd.to_datetime(self.all_trades['timestamp'], errors='coerce')
            
            self.all_trades = safe_concat(
                [self.all_trades, df],
                ignore_index=True
            )

            if 'trade_index' in self.all_trades.columns:
                self.all_trades = (
                    self.all_trades
                    .drop_duplicates(
                        subset=['trade_index'],
                        keep='last'
                    )
                )
                
                # Sort per timestamp
                if 'timestamp' in self.all_trades.columns:
                    try:
                        self.all_trades = self.all_trades.sort_values(
                            by="timestamp",
                            ascending=False
                        )
                    except Exception as e:
                        # Log errore se fallisce
                        self.logger.error(f"Failed to sort by timestamp: {e}")
                        # DEBUG: mostra tipi se fallisce
                        ts_types = self.all_trades['timestamp'].apply(type).value_counts()
                        self.logger.error(f"Timestamp types: {ts_types.to_dict()}")

        self.trade_table.update_data(self.all_trades)
        self._update_metrics()
        self._update_all_detached_pivots(self.all_trades)
        self._update_all_detached_charts(self.all_trades)

    def _on_filtered_data_changed(self, filtered_df: pd.DataFrame):
        """Callback su cambio filtri tabella."""
        self.current_filtered_data = filtered_df
        self._update_metrics()

    def _on_status_updated(self, status: dict):
        """Gestisce status update dal worker thread."""
        if status.get('type') == 'flow_detected':
            self._on_flow_detected(status.get('data', {}))
            return

        self.logger.debug(
            f"Status update received: {status.get('type')}"
        )

    def _on_flow_detected(self, flow_data: dict):
        """Gestisce evento flow_detected."""
        flow_id = flow_data.get('flow_id', 'unknown')
        self.logger.info(f"Flow received: {flow_id}")

        for flow_window in self.detached_flows:
            if not flow_window.isHidden():
                flow_window.add_flow(flow_data)

    def _on_error(self, error_msg: str):
        """Gestisce errori del worker thread."""
        self.logger.error(f"Worker thread error: {error_msg}")
        QMessageBox.warning(self, "Worker Error", error_msg)

    def _update_metrics(self):
        """Aggiorna le metriche nella dashboard"""
        if not self.metrics_enabled or self.all_trades.empty:
            return

        df = self.all_trades

        # Itera sugli item configurati
        for metric_key in self.metric_items:
            metric_def = METRIC_DEFINITIONS.get(metric_key)
            if not metric_def:
                self.logger.warning(f"Unknown metric key: {metric_key}")
                continue

            label = self.metric_labels.get(metric_key)
            if label is None:
                continue

            try:
                # Calcola valore
                value = metric_def["compute"](df)

                # Formatta
                fmt = metric_def.get("format", "{}")
                if isinstance(fmt, str):
                    text = fmt.format(value)
                else:
                    text = str(value)

                # Aggiorna label
                label.setText(f"{metric_def['label']}: {text}")

                # Colorizza se richiesto
                if metric_def.get("colorize", False) and isinstance(value, (int, float)):
                    if value > 0:
                        label.setStyleSheet("font-weight: bold; color: green; padding: 5px;")
                    elif value < 0:
                        label.setStyleSheet("font-weight: bold; color: red; padding: 5px;")
                    else:
                        label.setStyleSheet("font-weight: bold; padding: 5px;")
                else:
                    label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;")

            except Exception as e:
                self.logger.error(f"Error computing metric {metric_key}: {e}")
                label.setText(f"{metric_def['label']}: Error")
    # METODO clear_all() CORRETTO (sostituisci la sezione metriche)

    def clear_all(self):
        """Reset completo della dashboard."""
        self.logger.info("Clearing all dashboard data")

        self.all_trades = pd.DataFrame()
        self.current_filtered_data = pd.DataFrame()
        self.trade_table.clear()

        for pivot_window in self.detached_pivots:
            if not pivot_window.isHidden():
                pivot_window.clear_data()

        for chart_window in self.detached_charts:
            if not chart_window.isHidden():
                chart_window.chart_widget.set_data(pd.DataFrame())

        for flow_window in self.detached_flows:
            if not flow_window.isHidden():
                flow_window.clear_all()

        # Reset metriche
        if self.metrics_enabled:
            for label in self.metric_labels.values():
                # Trova il nome della metrica dalla label
                metric_name = "Metric"  # default
                for key, lbl in self.metric_labels.items():
                    if lbl == label:
                        metric_def = METRIC_DEFINITIONS.get(key)
                        if metric_def:
                            metric_name = metric_def['label']
                        break
                label.setText(f"{metric_name}: 0")
                label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;")

    def _show_column_chooser(self):
        """Mostra dialog per la selezione colonne."""
        if self.all_trades.empty:
            QMessageBox.information(
                self, "No Data", "No data available yet."
            )
            return

        available_columns = self.trade_table.get_available_columns()
        visible_columns = self.trade_table.get_visible_columns()

        if not available_columns:
            QMessageBox.information(
                self, "No Columns", "No columns available yet."
            )
            return

        dialog = ColumnChooserDialog(
            available_columns,
            visible_columns,
            self
        )
        if dialog.exec_() == QDialog.Accepted:
            selected_columns = dialog.get_selected_columns()
            if selected_columns:
                self.trade_table.set_visible_columns(selected_columns)
            else:
                QMessageBox.warning(
                    self,
                    "No Selection",
                    "Please select at least one column."
                )

    def _create_detached_pivot(self):
        """Crea una nuova finestra Pivot detached."""
        window = DetachedPivotWindow(
            self.all_trades,
            len(self.detached_pivots) + 1,
            self
        )
        window.data_update_needed.connect(
            self._update_single_detached_pivot
        )
        self.detached_pivots.append(window)
        window.show()
        self.logger.info(
            f"Created detached pivot window #{len(self.detached_pivots)}"
        )

    def _create_detached_chart(self):
        """Crea una nuova finestra Chart detached."""
        window = DetachedChartWindow(
            self.all_trades,
            len(self.detached_charts) + 1,
            self
        )
        self.detached_charts.append(window)
        window.show()
        self.logger.info(
            f"Created detached chart window #{len(self.detached_charts)}"
        )

    def _create_detached_flow(self):
        """Crea una nuova finestra Flow Monitor detached."""
        window = DetachedFlowWindow(
            len(self.detached_flows) + 1,
            self
        )
        self.detached_flows.append(window)
        window.show()
        self.logger.info(
            f"Created detached flow window #{len(self.detached_flows)}"
        )

    def _create_metrics_panel(self) -> QWidget:
        if not self.metrics_enabled:
            return QWidget()

        group = QGroupBox("Metrics Summary (All Data)")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        count = 0

        for key in self.metric_items:
            metric = METRIC_DEFINITIONS.get(key)
            if not metric:
                self.logger.warning(f"Unknown metric: {key}")
                continue

            label = QLabel(f"{metric['label']}: 0")
            label.setStyleSheet(
                "font-size: 13px; font-weight: bold; padding: 5px;"
            )

            self.metric_labels[key] = label
            row.addWidget(label)
            count += 1

            if count % 3 == 0:
                layout.addLayout(row)
                row = QHBoxLayout()

        if count % 3 != 0:
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _update_all_detached_pivots(self, data: pd.DataFrame):
        """Aggiorna tutte le finestre Pivot."""
        for window in self.detached_pivots:
            if not window.isHidden():
                window.update_source_data(data)

    def _update_all_detached_charts(self, data: pd.DataFrame):
        """Aggiorna tutte le finestre Chart."""
        for window in self.detached_charts:
            if not window.isHidden():
                window.update_source_data(data)

    def _update_single_detached_pivot(self, window):
        """Aggiorna una singola finestra Pivot."""
        if not window.isHidden():
            window.update_source_data(self.all_trades)

    def closeEvent(self, event):
        """Gestisce chiusura dashboard."""
        self.logger.info("Closing TradeDashboard")

        if self.worker_thread:
            self.logger.info("Stopping worker thread")
            self.worker_thread.stop()
            self.worker_thread.wait()

        for window in (
            self.detached_pivots +
            self.detached_charts +
            self.detached_flows
        ):
            window.close()

        super().closeEvent(event)


class ColumnChooserDialog(QDialog):
    """
    Dialog per la selezione delle colonne visibili
    nella tabella Raw Trades.
    """

    def __init__(self, available_columns, selected_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Columns")
        self.setModal(True)
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select columns to display:"))

        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(deselect_all_btn)

        layout.addLayout(btn_layout)

        self.list_widget = QListWidget()

        for col in available_columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if col in selected_columns else Qt.Unchecked
            )
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        self.count_label = QLabel()
        self._update_count()
        layout.addWidget(self.count_label)

        self.list_widget.itemChanged.connect(self._update_count)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def _deselect_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def _update_count(self):
        selected_count = len(self.get_selected_columns())
        total_count = self.list_widget.count()
        self.count_label.setText(
            f"Selected: {selected_count} / {total_count}"
        )

    def _on_accept(self):
        if len(self.get_selected_columns()) == 0:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one column."
            )
            return
        self.accept()

    def get_selected_columns(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected
