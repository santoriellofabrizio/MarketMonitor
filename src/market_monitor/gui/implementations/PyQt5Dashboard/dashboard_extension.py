"""  
TradeDashboard con integrazione DashboardState

Aggiungi questo codice alla tua TradeDashboard esistente
"""
import pandas as pd
from PyQt5.QtWidgets import (QMenu, QAction, QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QTextEdit, QDialogButtonBox,
                             QListWidget, QPushButton, QGroupBox, QMessageBox,
                             QInputDialog, QFileDialog, QListWidgetItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QIcon
from pathlib import Path

from market_monitor.gui.implementations.PyQt5Dashboard.widgets.dashboard_state import DashboardStateError
from market_monitor.gui.implementations.PyQt5Dashboard.detached_windows import DetachedChartWindow, DetachedFlowWindow, \
    DetachedPivotWindow


# ============================================================================
# MODIFICHE DA AGGIUNGERE A TradeDashboard
# ============================================================================

class TradeDashboardExtensions:
    """
    Mixin class con estensioni per TradeDashboard.

    COME USARE:
    1. Aggiungi questo import in trade_dashboard.py:
       from .DashboardExtensions import TradeDashboardExtensions

    2. Modifica la classe:
       class TradeDashboard(BasePyQt5Dashboard, TradeDashboardExtensions):

    3. Nel __init__() aggiungi:
       self.dashboard_state = DashboardState()
       self._setup_dashboard_menu()
    """

    def _setup_dashboard_menu(self):
        """Setup menu Dashboard - AGGIUNGERE nel __init__() dopo setup_ui()"""
        # Crea menu bar se non esiste
        menubar = self.menuBar()

        # Menu Dashboard
        dashboard_menu = menubar.addMenu("&Dashboard")

        # Save
        save_action = QAction(QIcon.fromTheme("document-save"), "&Save Dashboard...", self)
        save_action.setShortcut(QKeySequence.Save)  # Ctrl+S
        save_action.setStatusTip("Save current dashboard configuration")
        save_action.triggered.connect(self._save_dashboard_dialog)
        dashboard_menu.addAction(save_action)

        # Load
        load_action = QAction(QIcon.fromTheme("document-open"), "&Load Dashboard...", self)
        load_action.setShortcut(QKeySequence.Open)  # Ctrl+O
        load_action.setStatusTip("Load a saved dashboard configuration")
        load_action.triggered.connect(self._load_dashboard_dialog)
        dashboard_menu.addAction(load_action)

        dashboard_menu.addSeparator()

        # Save as Default
        save_default_action = QAction("Save as &Default", self)
        save_default_action.setStatusTip("Save current configuration as default")
        save_default_action.triggered.connect(self._save_as_default)
        dashboard_menu.addAction(save_default_action)

        # Reset to Default
        reset_action = QAction("&Reset to Default", self)
        reset_action.setStatusTip("Reset dashboard to default configuration")
        reset_action.triggered.connect(self._reset_to_default)
        dashboard_menu.addAction(reset_action)

        dashboard_menu.addSeparator()

        # Manage
        manage_action = QAction("&Manage Dashboards...", self)
        manage_action.setStatusTip("Manage saved dashboards")
        manage_action.triggered.connect(self._manage_dashboards_dialog)
        dashboard_menu.addAction(manage_action)

        dashboard_menu.addSeparator()

        # Export
        export_action = QAction(QIcon.fromTheme("document-save-as"), "&Export...", self)
        export_action.setStatusTip("Export dashboard to file")
        export_action.triggered.connect(self._export_dashboard_dialog)
        dashboard_menu.addAction(export_action)

        # Import
        import_action = QAction(QIcon.fromTheme("document-open"), "&Import...", self)
        import_action.setStatusTip("Import dashboard from file")
        import_action.triggered.connect(self._import_dashboard_dialog)
        dashboard_menu.addAction(import_action)

        # Auto-load default on startup
        self._try_load_default_dashboard()

    def _save_dashboard_dialog(self):
        """Dialog per salvare dashboard"""
        dialog = SaveDashboardDialog(self)

        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()

            if not values['name']:
                QMessageBox.warning(self, "Invalid Input", "Dashboard name is required.")
                return

            try:
                saved_file = self.dashboard_state.save_dashboard(
                    name=values['name'],
                    dashboard=self,
                    description=values['description'],
                    tags=values['tags']
                )

                QMessageBox.information(
                    self, "Success",
                    f"Dashboard '{values['name']}' saved successfully!\n\n"
                    f"Location: {saved_file}"
                )

            except DashboardStateError as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _load_dashboard_dialog(self):
        """Dialog per caricare dashboard"""
        dialog = LoadDashboardDialog(self.dashboard_state, self)

        if dialog.exec_() == QDialog.Accepted and dialog.selected_name:
            try:
                # Pulisci i dati vecchi prima di caricare
                old_shape = self.all_trades.shape if not self.all_trades.empty else (0, 0)
                self.all_trades = pd.DataFrame()
                self.current_filtered_data = pd.DataFrame()
                self.logger.info(f"Cleared all_trades (was {old_shape}) before loading dashboard")

                metadata = self.dashboard_state.load_dashboard(dialog.selected_name, self)

                # DEBUG: Verifica se all_trades è stato ripopolato
                if not self.all_trades.empty:
                    self.logger.warning(f"all_trades was repopulated during load! Shape: {self.all_trades.shape}")
                    if 'timestamp' in self.all_trades.columns:
                        ts_types = self.all_trades['timestamp'].apply(type).value_counts()
                        self.logger.warning(f"Timestamp types after load: {ts_types.to_dict()}")
                else:
                    self.logger.info("all_trades is empty after load (correct)")

                QMessageBox.information(
                    self, "Success",
                    f"Dashboard '{metadata['name']}' loaded successfully!"
                )

            except DashboardStateError as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _save_as_default(self):
        """Salva come default"""
        try:
            self.dashboard_state.save_dashboard("_default", self, description="Default dashboard")
            QMessageBox.information(self, "Success", "Default dashboard saved!")
        except DashboardStateError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _reset_to_default(self):
        """Reset a default"""
        reply = QMessageBox.question(
            self, "Confirm Reset",
            "Reset dashboard to default configuration?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Pulisci i dati vecchi prima di resettare
                self.all_trades = pd.DataFrame()
                self.current_filtered_data = pd.DataFrame()
                self.logger.info("Cleared old data before reset to default")

                self.dashboard_state.load_dashboard("_default", self)

                # Aggiorna se ci sono dati
                if not self.all_trades.empty:
                    self.trade_table.update_data(self.all_trades)
                    self.logger.info(f"Refreshed table with {len(self.all_trades)} trades after reset to default")

                QMessageBox.information(self, "Success", "Dashboard reset to default!")
            except DashboardStateError:
                QMessageBox.warning(self, "No Default", "No default dashboard found.")

    def _manage_dashboards_dialog(self):
        """Dialog per gestire dashboards"""
        dialog = ManageDashboardsDialog(self.dashboard_state, self)
        dialog.exec_()

    def _export_dashboard_dialog(self):
        """Dialog per esportare dashboard"""
        dialog = SaveDashboardDialog(self)
        dialog.setWindowTitle("Export Dashboard")

        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()

            if not values['name']:
                QMessageBox.warning(self, "Invalid Input", "Dashboard name is required.")
                return

            # Chiedi dove salvare
            export_path, _ = QFileDialog.getSaveFileName(
                self, "Export Dashboard",
                f"{values['name']}.json",
                "JSON Files (*.json)"
            )

            if export_path:
                try:
                    # Salva temporaneamente
                    self.dashboard_state.save_dashboard(
                        name=values['name'],
                        dashboard=self,
                        description=values['description'],
                        tags=values['tags']
                    )

                    # Esporta
                    if self.dashboard_state.export_dashboard(values['name'], Path(export_path)):
                        QMessageBox.information(self, "Success", f"Dashboard exported to:\n{export_path}")
                    else:
                        QMessageBox.critical(self, "Error", "Failed to export dashboard.")

                except DashboardStateError as e:
                    QMessageBox.critical(self, "Export Error", str(e))

    def _import_dashboard_dialog(self):
        """Dialog per importare dashboard"""
        import_path, _ = QFileDialog.getOpenFileName(
            self, "Import Dashboard",
            "",
            "JSON Files (*.json)"
        )

        if import_path:
            new_name, ok = QInputDialog.getText(
                self, "Import Dashboard",
                "Enter name for imported dashboard\n(leave empty to use original name):"
            )

            if ok:
                try:
                    if self.dashboard_state.import_dashboard(
                            Path(import_path),
                            new_name if new_name else None
                    ):
                        QMessageBox.information(self, "Success", "Dashboard imported successfully!")
                    else:
                        QMessageBox.critical(self, "Error", "Failed to import dashboard.")

                except DashboardStateError as e:
                    QMessageBox.critical(self, "Import Error", str(e))

    def _try_load_default_dashboard(self):
        """Try to load default dashboard configuration if it exists"""
        try:
            # Pulisci i dati vecchi prima di caricare il default
            self.all_trades = pd.DataFrame()
            self.current_filtered_data = pd.DataFrame()

            self.dashboard_state.load_dashboard("_default", self)

            # Aggiorna se ci sono dati
            if not self.all_trades.empty:
                self.trade_table.update_data(self.all_trades)
                self.logger.info(f"Refreshed table with {len(self.all_trades)} trades after loading default dashboard")

            print("✅ Default dashboard loaded")
        except DashboardStateError as e:
            # Dashboard non trovata - normale per prima esecuzione
            if "not found" in str(e).lower():
                print("ℹ️  No default dashboard - using empty configuration")
            else:
                print(f"⚠️  Could not load default dashboard: {e}")
        except FileNotFoundError:
            # File non esiste - normale per prima esecuzione
            print("ℹ️  No default dashboard - using empty configuration")
        except Exception as e:
            # Altro errore - log ma continua senza crashare
            print(f"⚠️  Unexpected error loading default dashboard: {e}")

    # Helper methods per detached windows (da aggiungere se non esistono)
    def _create_detached_pivot_internal(self):
        """Crea finestra pivot senza mostrare (per load)"""
        window_number = len(self.detached_pivots) + 1
        window = DetachedPivotWindow(self.current_filtered_data, window_number, self)
        self.detached_pivots.append(window)
        return window

    def _create_detached_chart_internal(self):
        """Crea finestra chart senza mostrare (per load)"""
        window_number = len(self.detached_charts) + 1
        window = DetachedChartWindow(self.current_filtered_data, window_number, self)
        self.detached_charts.append(window)
        return window

    def _create_detached_flow_internal(self):
        """Crea finestra flow senza mostrare (per load)"""
        window_number = len(self.detached_flows) + 1
        window = DetachedFlowWindow(window_number, self)
        self.detached_flows.append(window)
        return window


# ============================================================================
# DIALOGS
# ============================================================================

class SaveDashboardDialog(QDialog):
    """Dialog per salvare una dashboard"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Dashboard")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Name
        layout.addWidget(QLabel("<b>Dashboard Name:</b>"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., My Trading View")
        layout.addWidget(self.name_input)

        # Description
        layout.addWidget(QLabel("<b>Description (optional):</b>"))
        self.desc_input = QTextEdit()
        self.desc_input.setMaximumHeight(80)
        self.desc_input.setPlaceholderText("e.g., Dashboard with BTC charts and filters")
        layout.addWidget(self.desc_input)

        # Tags
        layout.addWidget(QLabel("<b>Tags (comma-separated, optional):</b>"))
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("e.g., trading, btc, analysis")
        layout.addWidget(self.tags_input)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        tags = [t.strip() for t in self.tags_input.text().split(',') if t.strip()]
        return {
            'name': self.name_input.text().strip(),
            'description': self.desc_input.toPlainText().strip(),
            'tags': tags
        }


class LoadDashboardDialog(QDialog):
    """Dialog per caricare una dashboard"""

    def __init__(self, dashboard_state, parent=None):
        super().__init__(parent)
        self.dashboard_state = dashboard_state
        self.selected_name = None
        self.setWindowTitle("Load Dashboard")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self._setup_ui()
        self._load_dashboards()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Select Dashboard:</b>"))

        # Lista dashboards
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.accept)
        self.list_widget.itemClicked.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # Info panel
        info_group = QGroupBox("Dashboard Info")
        info_layout = QVBoxLayout()

        self.info_label = QLabel("Select a dashboard to see details")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 10px;")
        info_layout.addWidget(self.info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_dashboards(self):
        self.list_widget.clear()
        self.dashboards = self.dashboard_state.list_dashboards()

        if not self.dashboards:
            self.list_widget.addItem("(No saved dashboards)")
            return

        for dashboard in self.dashboards:
            name = dashboard['name']
            timestamp = dashboard.get('timestamp', '')[:19]
            item_text = f"{name}  ({timestamp})"
            self.list_widget.addItem(item_text)

    def _on_selection_changed(self, item):
        if not self.dashboards:
            return

        idx = self.list_widget.row(item)
        dashboard = self.dashboards[idx]

        info = f"<b>Name:</b> {dashboard['name']}<br>"
        info += f"<b>Created:</b> {dashboard.get('timestamp', 'N/A')[:19]}<br>"

        if dashboard.get('description'):
            info += f"<b>Description:</b> {dashboard['description']}<br>"

        if dashboard.get('tags'):
            info += f"<b>Tags:</b> {', '.join(dashboard['tags'])}<br>"

        size_kb = dashboard.get('file_size', 0) / 1024
        info += f"<b>Size:</b> {size_kb:.1f} KB"

        self.info_label.setText(info)

    def _on_accept(self):
        if not self.dashboards:
            self.reject()
            return

        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.selected_name = self.dashboards[current_row]['name']
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a dashboard to load.")


class ManageDashboardsDialog(QDialog):
    """Dialog per gestire dashboards (delete, rename, duplicate)"""

    def __init__(self, dashboard_state, parent=None):
        super().__init__(parent)
        self.dashboard_state = dashboard_state
        self.setWindowTitle("Manage Dashboards")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self._setup_ui()
        self._load_dashboards()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Saved Dashboards:</b>"))

        # Lista
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # Info
        info_group = QGroupBox("Details")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("Select a dashboard")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_dashboard)
        btn_layout.addWidget(self.delete_btn)

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.setEnabled(False)
        self.rename_btn.clicked.connect(self._rename_dashboard)
        btn_layout.addWidget(self.rename_btn)

        self.duplicate_btn = QPushButton("Duplicate")
        self.duplicate_btn.setEnabled(False)
        self.duplicate_btn.clicked.connect(self._duplicate_dashboard)
        btn_layout.addWidget(self.duplicate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Close
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _load_dashboards(self):
        self.list_widget.clear()
        self.dashboards = self.dashboard_state.list_dashboards()

        if not self.dashboards:
            self.list_widget.addItem("(No saved dashboards)")
            return

        for dashboard in self.dashboards:
            self.list_widget.addItem(dashboard['name'])

    def _on_selection_changed(self, item):
        if not self.dashboards:
            return

        idx = self.list_widget.row(item)
        dashboard = self.dashboards[idx]

        info = f"<b>Name:</b> {dashboard['name']}<br>"
        info += f"<b>Created:</b> {dashboard.get('timestamp', 'N/A')[:19]}<br>"
        info += f"<b>Description:</b> {dashboard.get('description', 'N/A')}<br>"

        if dashboard.get('tags'):
            info += f"<b>Tags:</b> {', '.join(dashboard['tags'])}<br>"

        self.info_label.setText(info)

        self.delete_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)
        self.duplicate_btn.setEnabled(True)

    def _delete_dashboard(self):
        current_row = self.list_widget.currentRow()
        if current_row < 0:
            return

        dashboard_name = self.dashboards[current_row]['name']

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete dashboard '{dashboard_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.dashboard_state.delete_dashboard(dashboard_name):
                QMessageBox.information(self, "Success", "Dashboard deleted!")
                self._load_dashboards()
            else:
                QMessageBox.warning(self, "Error", "Failed to delete dashboard.")

    def _rename_dashboard(self):
        current_row = self.list_widget.currentRow()
        if current_row < 0:
            return

        old_name = self.dashboards[current_row]['name']

        new_name, ok = QInputDialog.getText(
            self, "Rename Dashboard",
            f"New name for '{old_name}':",
            text=old_name
        )

        if ok and new_name:
            if self.dashboard_state.rename_dashboard(old_name, new_name):
                QMessageBox.information(self, "Success", "Dashboard renamed!")
                self._load_dashboards()
            else:
                QMessageBox.warning(self, "Error", "Failed to rename dashboard.")

    def _duplicate_dashboard(self):
        current_row = self.list_widget.currentRow()
        if current_row < 0:
            return

        old_name = self.dashboards[current_row]['name']

        new_name, ok = QInputDialog.getText(
            self, "Duplicate Dashboard",
            f"Name for copy of '{old_name}':",
            text=f"{old_name} (copy)"
        )

        if ok and new_name:
            if self.dashboard_state.duplicate_dashboard(old_name, new_name):
                QMessageBox.information(self, "Success", "Dashboard duplicated!")
                self._load_dashboards()
            else:
                QMessageBox.warning(self, "Error", "Failed to duplicate dashboard.")
