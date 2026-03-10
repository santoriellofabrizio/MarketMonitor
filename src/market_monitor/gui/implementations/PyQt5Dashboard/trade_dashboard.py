"""
Dashboard completa per trades (PyQt5Dashboard) - CON WORKER THREAD

Dashboard PyQt5 per la visualizzazione real-time dei trade con supporto
a worker thread in background.

Supporta tre modalità di alimentazione dati:
- queue  : polling da Queue interna allo stesso processo Strategy
- redis  : sottoscrizione Redis Pub/Sub da processo separato
- rabbit : sottoscrizione RabbitMQ fanout exchange da processo separato

Caratteristiche:
- Tabella Raw Trades con deduplicazione e filtri
- Metriche aggregate globali (calcolate su ALL data)
- Finestre detachable per Pivot, Chart e Flow Monitor
- Separazione rigorosa tra worker thread e UI thread
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QDialog, QListWidget,
    QDialogButtonBox, QListWidgetItem, QMessageBox,
    QTableWidget, QTableWidgetItem, QCheckBox, QComboBox, QHeaderView,
    QAbstractItemView,
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
    DetachedFlowWindow,
    DetachedGroupByWindow,
)
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.dashboard_state import DashboardState
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.trade_table import TradeTableWidget
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.trade_history_window import TradeHistoryWindow
from market_monitor.gui.threaded_GUI.QueueDataSource import QueueDataSource
from market_monitor.gui.implementations.PyQt5Dashboard.worker_thread import (
    QueuePollingThread,
    RedisPubSubThread,
    RabbitPubSubThread,
)


DEFAULT_METRIC_ITEMS = [
    {"label": "Total Trades",    "expr": "spread_pl",   "aggregation": "count", "colorize": False, "format": ""},
    {"label": "Spread P&L",      "expr": "spread_pl",   "aggregation": "sum",   "colorize": True,  "format": "€{:,.2f}"},
    {"label": "Avg Marginality", "expr": "marginality", "aggregation": "mean",  "colorize": True,  "format": "{:.2%}"},
]

_AGGREGATIONS = ["count", "sum", "mean", "median", "min", "max", "std", "last"]


class MetricDefinitionDialog(QDialog):
    """
    Dialog to define metric summary items dynamically.

    Each metric is defined by:
    - label:       display name (auto-filled if blank)
    - expr:        column name or pandas eval expression (e.g. 'spread_pl / ctv')
    - aggregation: one of count/sum/mean/median/min/max/std/last
    - colorize:    bool — green/red for positive/negative values
    - format:      optional Python format string (e.g. '€{:,.2f}')
    """

    def __init__(self, metric_items: list, data_columns: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Metric Summary")
        self.setMinimumSize(700, 400)
        self.data_columns = data_columns

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Define metrics: choose a column (or type a pandas eval expression) and an aggregation."
        ))

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Label", "Column / Expression", "Aggregation", "Colorize", "Format"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        # Buttons row
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_empty_row)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("− Remove")
        remove_btn.clicked.connect(self._remove_selected_row)
        btn_row.addWidget(remove_btn)

        up_btn = QPushButton("↑ Up")
        up_btn.clicked.connect(self._move_row_up)
        btn_row.addWidget(up_btn)

        down_btn = QPushButton("↓ Down")
        down_btn.clicked.connect(self._move_row_down)
        btn_row.addWidget(down_btn)

        btn_row.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        for item in metric_items:
            if isinstance(item, dict):
                self._add_row(item)

    def _make_expr_combo(self, current_expr: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems([""] + self.data_columns)
        combo.setCurrentText(current_expr)
        return combo

    def _make_agg_combo(self, current_agg: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(_AGGREGATIONS)
        idx = combo.findText(current_agg)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        return combo

    def _make_colorize_widget(self, checked: bool) -> QWidget:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch()
        chk = QCheckBox()
        chk.setChecked(checked)
        h.addWidget(chk)
        h.addStretch()
        return container

    def _add_row(self, item: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(item.get("label", "")))
        self.table.setCellWidget(row, 1, self._make_expr_combo(item.get("expr", "")))
        self.table.setCellWidget(row, 2, self._make_agg_combo(item.get("aggregation", "sum")))
        self.table.setCellWidget(row, 3, self._make_colorize_widget(bool(item.get("colorize", False))))
        self.table.setItem(row, 4, QTableWidgetItem(item.get("format", "")))

    def _add_empty_row(self):
        self._add_row({"label": "", "expr": "", "aggregation": "sum", "colorize": False, "format": ""})

    def _remove_selected_row(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _move_row_up(self):
        row = self.table.currentRow()
        if row > 0:
            self._swap_rows(row, row - 1)
            self.table.setCurrentCell(row - 1, self.table.currentColumn())

    def _move_row_down(self):
        row = self.table.currentRow()
        if row < self.table.rowCount() - 1:
            self._swap_rows(row, row + 1)
            self.table.setCurrentCell(row + 1, self.table.currentColumn())

    def _swap_rows(self, r1: int, r2: int):
        """Swap two rows by reading all widgets/items and re-setting them."""
        def _read_row(r):
            label = (self.table.item(r, 0).text() if self.table.item(r, 0) else "")
            expr = self.table.cellWidget(r, 1).currentText()
            agg = self.table.cellWidget(r, 2).currentText()
            colorize = self.table.cellWidget(r, 3).findChild(QCheckBox).isChecked()
            fmt = (self.table.item(r, 4).text() if self.table.item(r, 4) else "")
            return {"label": label, "expr": expr, "aggregation": agg, "colorize": colorize, "format": fmt}

        d1 = _read_row(r1)
        d2 = _read_row(r2)
        self._set_row(r1, d2)
        self._set_row(r2, d1)

    def _set_row(self, row: int, item: dict):
        self.table.setItem(row, 0, QTableWidgetItem(item["label"]))
        self.table.setCellWidget(row, 1, self._make_expr_combo(item["expr"]))
        self.table.setCellWidget(row, 2, self._make_agg_combo(item["aggregation"]))
        self.table.setCellWidget(row, 3, self._make_colorize_widget(item["colorize"]))
        self.table.setItem(row, 4, QTableWidgetItem(item["format"]))

    def get_metric_items(self) -> list:
        """Return list of metric dicts, skipping rows with empty expression."""
        items = []
        for row in range(self.table.rowCount()):
            expr = self.table.cellWidget(row, 1).currentText().strip()
            if not expr:
                continue
            label_item = self.table.item(row, 0)
            label = label_item.text().strip() if label_item else ""
            agg = self.table.cellWidget(row, 2).currentText()
            colorize = self.table.cellWidget(row, 3).findChild(QCheckBox).isChecked()
            fmt_item = self.table.item(row, 4)
            fmt = fmt_item.text().strip() if fmt_item else ""
            if not label:
                label = f"{expr} ({agg})"
            items.append({"label": label, "expr": expr, "aggregation": agg, "colorize": colorize, "format": fmt})
        return items


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
            rabbit_config: Optional[Dict] = None,
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
        self.rabbit_config = rabbit_config or {}
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
        raw_items = self.metrics_config.get("items", [])

        # Accept only new-style List[Dict]; discard old List[str] keys
        if raw_items and all(isinstance(x, dict) for x in raw_items):
            self.metric_items = raw_items
        else:
            self.metric_items = list(DEFAULT_METRIC_ITEMS)

        self.metric_labels = {}  # keyed by int index

        # ===== CAMPI CALCOLATI =====
        self.calculated_fields: Dict[str, str] = {}  # {'margin': 'spread_pl / ctv'}

        # ===== DETACHED WINDOWS =====
        self.detached_pivots = []
        self.detached_charts = []
        self.detached_flows = []
        self.detached_groupbys = []
        self.trade_history_windows = []

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
        self.logger.info(
            f"Starting TradeDashboard — mode={self.mode}, "
            f"redis={self.redis_config}, rabbit_host={self.rabbit_config.get('host')}"
        )
        self.show()
        self.start_monitoring()
        self._update_status_bar()
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

    def _update_status_bar(self):
        """Aggiorna la status bar con info config + contatori."""
        state = "PAUSED" if self.paused else "LIVE"
        total = len(self.all_trades) if not self.all_trades.empty else 0

        if self.mode == 'redis':
            host = self.redis_config.get('host', 'localhost')
            port = self.redis_config.get('port', 6379)
            channel = self.redis_config.get('channel', 'trades_df')
            source_info = f"Redis  {host}:{port}  channel={channel}"
        elif self.mode == 'rabbit':
            host = self.rabbit_config.get('host', '?')
            exchange = self.rabbit_config.get('exchange', '?')
            source_info = f"RabbitMQ  {host}  exchange={exchange}"
        elif self.mode == 'queue':
            source_info = "Queue (in-process)"
        else:
            source_info = f"mode={self.mode}"

        self.status_bar.showMessage(
            f"  {state}  |  {source_info}  |  trades={total}  |  "
            f"calc_fields={len(self.calculated_fields)}  |  metrics={len(self.metric_items)}"
        )

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

        elif self.mode == 'rabbit':
            self.worker_thread = RabbitPubSubThread(self.rabbit_config)

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

        clear_btn = QPushButton("🗑️ Clear Display")
        clear_btn.setToolTip("Clears chart/table rendering. Data is kept in memory and will re-appear on next update.")
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

        detach_groupby_btn = QPushButton("🪟 New GroupBy Window")
        detach_groupby_btn.clicked.connect(self._create_detached_groupby)
        layout.addWidget(detach_groupby_btn)

        if self.metrics_enabled:
            metrics_btn = QPushButton("⚙️ Metrics")
            metrics_btn.setToolTip("Choose which metrics to display in the summary panel")
            metrics_btn.clicked.connect(self._show_metric_chooser)
            layout.addWidget(metrics_btn)

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
        self._update_status_bar()

    def _on_data_received(self, df: pd.DataFrame):
        """
        Ricezione nuovi dati dal worker thread.
        Esegue SEMPRE nel main thread Qt.
        """
        if self.paused:
            return

        if df is None or df.empty:
            self.logger.debug("_on_data_received: empty or None DataFrame, skipping")
            return

        self.logger.debug(
            f"_on_data_received: {len(df)} rows, cols={list(df.columns)}, "
            f"all_trades_shape={self.all_trades.shape}"
        )

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
                # Sort ascending so the most recent row is last in each group,
                # then groupby.last() keeps the last non-null value per column.
                # This prevents losing horizon PL values when a later partial update arrives.
                sort_cols = (
                    ['timestamp', 'trade_index']
                    if 'timestamp' in self.all_trades.columns
                    else ['trade_index']
                )
                try:
                    self.all_trades = self.all_trades.sort_values(
                        by=sort_cols, ascending=True, na_position='first'
                    )
                except Exception as e:
                    self.logger.error(f"Failed to sort trades: {e}")

                self.all_trades = (
                    self.all_trades
                    .groupby('trade_index', sort=False)
                    .last()
                    .reset_index()
                )

                # Re-sort descending for display
                if 'timestamp' in self.all_trades.columns:
                    try:
                        self.all_trades = self.all_trades.sort_values(
                            by=["timestamp", "trade_index"],
                            ascending=False
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to sort by timestamp: {e}")
                        ts_types = self.all_trades['timestamp'].apply(type).value_counts()
                        self.logger.error(f"Timestamp types: {ts_types.to_dict()}")

        try:
            enriched = self._get_enriched_trades()
            self.logger.debug(
                f"Enriched trades: {len(enriched)} rows, "
                f"calculated_fields={list(self.calculated_fields.keys())}"
            )
            self.trade_table.update_data(enriched)
            self._update_metrics()
            self._update_all_detached_pivots(enriched)
            self._update_all_detached_charts(enriched)
            self._update_all_detached_groupbys(enriched)
            self._update_status_bar()
        except Exception as e:
            import traceback as _tb
            self.logger.error(f"_on_data_received: error updating UI: {e}\n{_tb.format_exc()}")

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
        
        # Assicura che flow_data abbia i tipi corretti per la GUI
        # Converte trades a lista vuota se è None
        if 'trades' not in flow_data or flow_data.get('trades') is None:
            flow_data['trades'] = []
        
        # Converte valori numerici
        for key in ['ctv', 'avg_interval', 'duration', 'consistency_score', 'avg_quantity', 'total_quantity', 'avg_price']:
            if key in flow_data:
                try:
                    flow_data[key] = float(flow_data[key]) if flow_data[key] is not None else 0.0
                except (ValueError, TypeError):
                    flow_data[key] = 0.0
        
        # Assicura che side sia string
        if 'side' in flow_data and flow_data['side'] is None:
            flow_data['side'] = 'UNKNOWN'

        for flow_window in self.detached_flows:
            if not flow_window.isHidden():
                # ✅ CONNESSIONE DINAMICA DEL SEGNALE
                # Questo assicura che il segnale sia collegato anche per finestre pre-salvate
                try:
                    flow_window.flow_widget.flow_selected.disconnect()
                except TypeError:
                    # Se la connessione non esiste, disconnect() lancia TypeError
                    pass
                
                flow_window.flow_widget.flow_selected.connect(self._show_trade_history)
                flow_window.add_flow(flow_data)

    def _on_error(self, error_msg: str):
        """Gestisce errori del worker thread — mostra nella status bar senza bloccare il flusso."""
        self.logger.error(f"Worker thread error: {error_msg}")
        # Mostra nella status bar invece di una modal dialog che blocca l'update loop
        self.status_bar.showMessage(f"⚠ Error: {error_msg}", 8000)

    def _update_metrics(self):
        """Aggiorna le metriche nella dashboard usando definizioni dinamiche.

        Usa i dati arricchiti (all_trades + campi calcolati) in modo che
        le metriche possano riferirsi anche a campi calcolati.
        """
        if not self.metrics_enabled or self.all_trades.empty:
            return

        df = self._get_enriched_trades()

        for idx, metric_def in enumerate(self.metric_items):
            if not isinstance(metric_def, dict):
                continue
            label_widget = self.metric_labels.get(idx)
            if label_widget is None:
                continue

            expr = metric_def.get("expr", "")
            agg = metric_def.get("aggregation", "sum")
            display_label = metric_def.get("label", expr)

            try:
                series = df[expr] if expr in df.columns else df.eval(expr)
                value = len(series.dropna()) if agg == "count" else getattr(series, agg)()
            except Exception as e:
                self.logger.warning(f"[Metric] '{display_label}' error: {e}")
                label_widget.setText(f"{display_label}: Error")
                continue

            fmt = metric_def.get("format", "")
            if fmt:
                try:
                    text = fmt.format(value)
                except Exception:
                    text = str(value)
            elif agg == "count" or isinstance(value, int):
                text = str(int(value))
            else:
                try:
                    text = f"{value:,.2f}"
                except Exception:
                    text = str(value)

            label_widget.setText(f"{display_label}: {text}")

            if metric_def.get("colorize") and isinstance(value, (int, float)):
                color = "green" if value > 0 else ("red" if value < 0 else "black")
                label_widget.setStyleSheet(f"font-weight: bold; color: {color}; padding: 5px;")
            else:
                label_widget.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;")
    # METODO clear_all() CORRETTO (sostituisci la sezione metriche)

    def clear_all(self):
        """Pulisce il rendering di tutti i widget. I dati restano in memoria e
        ricompariranno al prossimo aggiornamento dati."""
        self.logger.info("Clearing dashboard display (data retained in memory)")

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

        for groupby_window in self.detached_groupbys:
            if not groupby_window.isHidden():
                groupby_window.clear_data()

        # Reset metriche
        if self.metrics_enabled:
            for idx, metric_def in enumerate(self.metric_items):
                lbl = self.metric_labels.get(idx)
                if lbl and isinstance(metric_def, dict):
                    lbl.setText(f"{metric_def.get('label', '—')}: —")
                    lbl.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;")

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
        # ✅ Collega il signal dei flow a _show_trade_history
        window.flow_widget.flow_selected.connect(self._show_trade_history)
        
        self.detached_flows.append(window)
        window.show()
        self.logger.info(
            f"Created detached flow window #{len(self.detached_flows)}"
        )

    def _show_metric_chooser(self):
        """Apre il dialog per definire le metriche e ricostruisce il pannello.

        Le colonne disponibili includono sia quelle raw che i campi calcolati.
        """
        enriched = self._get_enriched_trades()
        columns = list(enriched.columns) if not enriched.empty else []
        # Se non ci sono dati ma ci sono campi calcolati, aggiungili comunque
        if not columns and self.calculated_fields:
            columns = list(self.calculated_fields.keys())
        dialog = MetricDefinitionDialog(
            metric_items=list(self.metric_items),
            data_columns=columns,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            self._rebuild_metrics_panel(dialog.get_metric_items())

    def _rebuild_metrics_panel(self, new_items: list):
        """Sostituisce il pannello metriche con uno nuovo basato su new_items."""
        self.metric_items = new_items
        self.metric_labels.clear()

        # Rimuovi e distruggi il vecchio pannello
        old_panel = getattr(self, 'metrics_panel', None)
        if old_panel is not None:
            self.main_layout.removeWidget(old_panel)
            old_panel.deleteLater()

        # Crea il nuovo pannello e inseriscilo alla posizione 1 (dopo controls)
        self.metrics_panel = self._create_metrics_panel()
        self.main_layout.insertWidget(1, self.metrics_panel)

        # Aggiorna subito con i dati correnti
        if not self.all_trades.empty:
            self._update_metrics()

    def _create_metrics_panel(self) -> QWidget:
        if not self.metrics_enabled:
            return QWidget()

        group = QGroupBox("Metrics Summary (All Data)")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        count = 0

        self.metric_labels = {}
        for idx, metric_def in enumerate(self.metric_items):
            if not isinstance(metric_def, dict):
                continue

            display_label = metric_def.get("label", f"metric_{idx}")
            label = QLabel(f"{display_label}: —")
            label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;")

            self.metric_labels[idx] = label
            row.addWidget(label)
            count += 1

            if count % 4 == 0:
                layout.addLayout(row)
                row = QHBoxLayout()

        if count % 4 != 0:
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _get_enriched_trades(self) -> pd.DataFrame:
        """Ritorna all_trades arricchito con i campi calcolati definiti dall'utente."""
        if self.all_trades.empty or not self.calculated_fields:
            return self.all_trades

        df = self.all_trades.copy()
        for name, expr in self.calculated_fields.items():
            try:
                df[name] = df.eval(expr)
            except Exception as e:
                self.logger.warning(f"[CalcField] '{name}' error: {e}")
        return df

    def _create_detached_groupby(self):
        """Crea una nuova finestra GroupBy detached."""
        window = DetachedGroupByWindow(
            self._get_enriched_trades(),
            len(self.detached_groupbys) + 1,
            self
        )
        self.detached_groupbys.append(window)
        window.show()
        self.logger.info(f"Created detached groupby window #{len(self.detached_groupbys)}")

    def _update_all_detached_groupbys(self, data: pd.DataFrame):
        """Aggiorna tutte le finestre GroupBy."""
        for window in self.detached_groupbys:
            if not window.isHidden():
                window.update_source_data(data)

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

    def _show_trade_history(self, flow_id: str):
        """Mostra la storia dei trades per il ticker del flow."""
        self.logger.info(f"=== _show_trade_history called ===")
        self.logger.info(f"flow_id: {flow_id}")
        self.logger.info(f"all_trades shape: {self.all_trades.shape}")
        self.logger.debug(f"all_trades columns: {list(self.all_trades.columns)}")
        
        # Find the ticker from the flow_id
        ticker = None
        
        for flow_window in self.detached_flows:
            if flow_window.isHidden():
                self.logger.debug(f"Flow window is hidden, skipping")
                continue
            
            self.logger.debug(f"Searching in flow_window, flow_cards count: {len(flow_window.flow_widget.flow_cards)}")
            
            for card in flow_window.flow_widget.flow_cards.values():
                self.logger.debug(f"Checking card with flow_id={card.flow_id}")
                
                if card.flow_id == flow_id:
                    # Try multiple possible keys for ticker
                    ticker = (
                        card.flow_data.get('ticker') or 
                        card.flow_data.get('instrument_id') or 
                        card.flow_data.get('isin')
                    )
                    
                    self.logger.info(f"Found flow {flow_id}, ticker={ticker}")
                    self.logger.debug(f"flow_data keys: {list(card.flow_data.keys())}")
                    break
            
            if ticker:
                break
        
        if not ticker:
            self.logger.warning(f"Could not find ticker for flow {flow_id}")
            QMessageBox.warning(self, "Flow Not Found", f"Could not find ticker for flow {flow_id}")
            return
        
        # Verify we have trade data
        if self.all_trades.empty:
            self.logger.warning("all_trades is empty - no data to display")
            QMessageBox.warning(self, "No Data", "No trade data available yet")
            return
        
        # Check if trades exist for this ticker
        if 'ticker' not in self.all_trades.columns:
            self.logger.error("'ticker' column not found in all_trades")
            QMessageBox.warning(self, "Missing Data", "Trades do not have 'ticker' column")
            return
        
        ticker_trades = self.all_trades[
            self.all_trades['ticker'].str.upper() == ticker.upper()
        ]
        
        if ticker_trades.empty:
            self.logger.warning(f"No trades found for ticker {ticker}")
            available_tickers = self.all_trades['ticker'].unique().tolist()
            self.logger.debug(f"Available tickers: {available_tickers}")
            QMessageBox.warning(self, "No Trades", f"No trades found for {ticker}")
            return
        
        # Create and show Trade History window
        self.logger.info(f"Opening Trade History for {ticker} ({len(ticker_trades)} trades)")
        
        history_window = TradeHistoryWindow(ticker, self.all_trades)
        history_window.show()
        
        # Keep reference to prevent garbage collection
        self.trade_history_windows.append(history_window)

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
            self.detached_flows +
            self.detached_groupbys
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

        layout.addWidget(QLabel(
            "Select columns to display. Drag items to reorder."
        ))

        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(deselect_all_btn)

        layout.addLayout(btn_layout)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)

        # Mostra prima le colonne selezionate (nell'ordine corrente), poi le restanti
        selected_set = set(selected_columns)
        ordered = list(selected_columns) + [c for c in available_columns if c not in selected_set]
        for col in ordered:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)
            item.setCheckState(
                Qt.Checked if col in selected_set else Qt.Unchecked
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
