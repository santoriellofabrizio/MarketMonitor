import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox

from market_monitor.gui.implementations.PyQt5Dashboard.widgets.chart_widget import ChartWidget
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.flow_monitor_widget import FlowMonitorWidget
from market_monitor.gui.implementations.PyQt5Dashboard.widgets.pivot_table import PivotTableWidget


class DetachedPivotWindow(QMainWindow):
    """Finestra separata per pivot table con aggiornamento dati"""

    # Signal per richiedere aggiornamento dati
    data_update_needed = pyqtSignal(object)  # Passa self

    def __init__(self, source_data, window_number, parent=None):
        super().__init__(parent)
        self.window_number = window_number
        self.setWindowTitle(f"Pivot Table #{window_number}")
        self.setGeometry(200 + (window_number * 30), 200 + (window_number * 30), 900, 600)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel(f"Detached Pivot Window #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e0f7fa;")
        layout.addWidget(self.info_label)

        # Pivot widget
        self.pivot_widget = PivotTableWidget()
        self.pivot_widget.set_source_data(source_data)
        layout.addWidget(self.pivot_widget)

    def update_source_data(self, data: pd.DataFrame):
        """Aggiorna i dati sorgente e ri-applica pivot se configurato"""
        self.pivot_widget.set_source_data(data)
        self.info_label.setText(
            f"Detached Pivot Window #{self.window_number} | "
            f"Source rows: {len(data)}"
        )

    def clear_data(self):
        """Pulisce i dati"""
        self.pivot_widget.clear_pivot()
        self.pivot_widget.set_source_data(pd.DataFrame())


class DetachedChartWindow(QMainWindow):
    """Finestra separata per grafici"""

    def __init__(self, source_data, window_number, parent=None):
        super().__init__(parent)
        self.window_number = window_number
        self.setWindowTitle(f"Charts #{window_number}")
        self.setGeometry(300 + (window_number * 30), 300 + (window_number * 30), 1000, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel(f"Detached Chart Window #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #fff3e0;")
        layout.addWidget(self.info_label)

        # Chart widget
        self.chart_widget = ChartWidget()
        self.chart_widget.set_data(source_data)
        layout.addWidget(self.chart_widget)

    def update_source_data(self, data: pd.DataFrame):
        """Aggiorna i dati sorgente (auto-update se abilitato)"""
        self.chart_widget.set_data(data)
        self.info_label.setText(
            f"Detached Chart Window #{self.window_number} | "
            f"Source rows: {len(data)}"
        )


class DetachedFlowWindow(QMainWindow):
    """Finestra separata per Flow Monitor"""

    def __init__(self, window_number, parent=None):
        super().__init__(parent)
        self.window_number = window_number
        self.setWindowTitle(f"Flow Monitor #{window_number}")
        self.setGeometry(400 + (window_number * 30), 400 + (window_number * 30), 500, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel(f"Flow Monitor #{window_number}")
        self.info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #e3f2fd;")
        layout.addWidget(self.info_label)

        # Flow Monitor widget (sempre visibile in finestra detached)
        self.flow_widget = FlowMonitorWidget(auto_hide=False)
        layout.addWidget(self.flow_widget)

        # NON connettere qui - sarà connesso dalla dashboard a _show_trade_history
        # self.flow_widget.flow_selected.connect(self._on_flow_selected)

    def add_flow(self, flow_data: dict):
        """Aggiungi un flow al monitor"""
        self.flow_widget.update_flows([flow_data])

        # Aggiorna contatore nel titolo
        flow_count = len(self.flow_widget.flow_cards)
        self.info_label.setText(
            f"Flow Monitor #{self.window_number} | "
            f"Flows: {flow_count}"
        )

    def clear_all(self):
        """Pulisce tutti i flow"""
        self.flow_widget.clear_all()
        self.info_label.setText(f"Flow Monitor #{self.window_number} | Flows: 0")