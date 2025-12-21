"""
FlowMonitorWidget - Displays detected trading flows in real-time.
Compact design with instrument filtering.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QScrollArea, QFrame, QPushButton, QGroupBox,
                             QCheckBox, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import List, Dict, Set


class FlowCard(QFrame):
    """Compact card widget to display a single flow."""

    clicked = pyqtSignal(str)  # flow_id

    def __init__(self, flow_data: dict, parent=None):
        super().__init__(parent)
        self.flow_id = flow_data['flow_id']
        self.flow_data = flow_data

        self._setup_ui()
        self._update_data(flow_data)

        self.setCursor(Qt.PointingHandCursor)

    def _setup_ui(self):
        """Setup compact card UI."""
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(1)
        self.setMaximumHeight(85)  # Compact height

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(3)

        # --- Header: Instrument ID + Flow ID ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        # Instrument ID (BOLD, larger)
        self.instrument_label = QLabel()
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.instrument_label.setFont(font)
        header_layout.addWidget(self.instrument_label)

        # Flow ID (smaller, gray)
        self.flow_id_label = QLabel()
        self.flow_id_label.setStyleSheet("font-size: 9px; color: #666;")
        header_layout.addWidget(self.flow_id_label)

        # Side badge
        self.side_label = QLabel()
        self.side_label.setStyleSheet(
            "font-size: 9px; font-weight: bold; "
            "padding: 2px 6px; border-radius: 3px;"
        )
        header_layout.addWidget(self.side_label)

        # Status indicator
        self.status_label = QLabel()
        header_layout.addWidget(self.status_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # --- Metrics Grid (2 rows x 3 cols) ---
        metrics_container = QWidget()
        metrics_layout = QHBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(10)

        # Column 1
        col1 = QVBoxLayout()
        col1.setSpacing(0)
        self.trades_label = QLabel()
        self.trades_label.setStyleSheet("font-size: 9px;")
        col1.addWidget(self.trades_label)

        self.ctv_label = QLabel()
        self.ctv_label.setStyleSheet("font-size: 9px;")
        col1.addWidget(self.ctv_label)
        metrics_layout.addLayout(col1)

        # Column 2
        col2 = QVBoxLayout()
        col2.setSpacing(0)
        self.interval_label = QLabel()
        self.interval_label.setStyleSheet("font-size: 9px;")
        col2.addWidget(self.interval_label)

        self.duration_label = QLabel()
        self.duration_label.setStyleSheet("font-size: 9px;")
        col2.addWidget(self.duration_label)
        metrics_layout.addLayout(col2)

        # Column 3
        col3 = QVBoxLayout()
        col3.setSpacing(0)
        self.consistency_label = QLabel()
        self.consistency_label.setStyleSheet("font-size: 9px;")
        col3.addWidget(self.consistency_label)

        self.avg_qty_label = QLabel()
        self.avg_qty_label.setStyleSheet("font-size: 9px;")
        col3.addWidget(self.avg_qty_label)
        metrics_layout.addLayout(col3)

        metrics_layout.addStretch()
        layout.addWidget(metrics_container)

    def _update_data(self, flow_data: dict):
        """Update card with flow data."""
        self.flow_data = flow_data

        # --- Header ---
        # Instrument ID
        instrument_id = flow_data.get('instrument_id', flow_data.get('ticker', 'N/A'))
        self.instrument_label.setText(instrument_id)

        # Flow ID
        self.flow_id_label.setText(f"[{flow_data['flow_id']}]")

        # Side badge
        side = flow_data['side'].upper()
        if side in ['BUY', 'BID']:
            self.side_label.setText("BUY")
            self.side_label.setStyleSheet(
                "font-size: 9px; font-weight: bold; "
                "background-color: #4caf50; color: white; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            self._set_color('#e8f5e9')  # Very light green
        else:
            self.side_label.setText("SELL")
            self.side_label.setStyleSheet(
                "font-size: 9px; font-weight: bold; "
                "background-color: #f44336; color: white; "
                "padding: 2px 6px; border-radius: 3px;"
            )
            self._set_color('#ffebee')  # Very light red

        # Status
        if flow_data['is_active']:
            self.status_label.setText("🟢")
        else:
            self.status_label.setText("⚪")

        # --- Metrics ---
        # Row 1
        self.trades_label.setText(f"Trades: {len(flow_data.get('trades', []))}")

        ctv = flow_data.get('ctv', flow_data.get('total_quantity', 0) * flow_data.get('avg_price', 0))
        self.ctv_label.setText(f"CTV: {ctv:,.0f}")

        # Row 2
        self.interval_label.setText(f"Int: {flow_data['avg_interval']:.1f}s")
        self.duration_label.setText(f"Dur: {flow_data['duration']:.1f}s")

        # Consistency with color
        score = flow_data['consistency_score']
        if score > 0.7:
            color = '#4caf50'  # Green
        elif score > 0.4:
            color = '#ff9800'  # Orange
        else:
            color = '#f44336'  # Red

        self.consistency_label.setText(
            f"<span style='color: {color}; font-weight: bold;'>Cons: {score:.2f}</span>"
        )

        self.avg_qty_label.setText(f"Qty: {flow_data['avg_quantity']:.0f}")

    def _set_color(self, color: str):
        """Set card background color."""
        self.setStyleSheet(f"""
            FlowCard {{
                background-color: {color};
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
            FlowCard:hover {{
                border: 2px solid #333;
                background-color: #fff9c4;
            }}
        """)

    def update_flow(self, flow_data: dict):
        """Update card with new flow data."""
        self._update_data(flow_data)

    def mousePressEvent(self, event):
        """Handle click."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.flow_id)
        super().mousePressEvent(event)


class FlowMonitorWidget(QWidget):
    """
    Widget to monitor and display detected flows.
    Compact design with instrument filtering.
    """

    flow_selected = pyqtSignal(str)  # flow_id

    def __init__(self, parent=None, auto_hide=True):
        super().__init__(parent)

        self.flow_cards: Dict[str, FlowCard] = {}  # flow_id -> FlowCard
        self.auto_hide = auto_hide
        self.has_ever_had_flows = False

        # Filtering
        self.hidden_instruments: Set[str] = set()  # Instruments to hide

        self._setup_ui()

        if self.auto_hide:
            self.hide()

    def _setup_ui(self):
        """Setup UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # --- Header ---
        header = QGroupBox("Flow Monitor")
        header_layout = QVBoxLayout()
        header_layout.setSpacing(5)

        # Top row: Count + Buttons
        top_row = QHBoxLayout()

        self.count_label = QLabel("Active: 0")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        top_row.addWidget(self.count_label)

        top_row.addStretch()

        # Clear All button
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setMaximumWidth(80)
        clear_all_btn.clicked.connect(self.clear_all)
        clear_all_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "padding: 3px 8px; border-radius: 3px; font-size: 9px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
        )
        top_row.addWidget(clear_all_btn)

        # Clear Completed button
        clear_btn = QPushButton("Clear Done")
        clear_btn.setMaximumWidth(80)
        clear_btn.clicked.connect(self._clear_completed)
        clear_btn.setStyleSheet(
            "QPushButton { background-color: #757575; color: white; "
            "padding: 3px 8px; border-radius: 3px; font-size: 9px; }"
            "QPushButton:hover { background-color: #616161; }"
        )
        top_row.addWidget(clear_btn)

        header_layout.addLayout(top_row)

        # Bottom row: Filter
        filter_row = QHBoxLayout()

        filter_label = QLabel("Hide:")
        filter_label.setStyleSheet("font-size: 9px;")
        filter_row.addWidget(filter_label)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter instrument ID to hide...")
        self.filter_input.setMaximumHeight(22)
        self.filter_input.setStyleSheet("font-size: 9px;")
        filter_row.addWidget(self.filter_input)

        add_filter_btn = QPushButton("Hide")
        add_filter_btn.setMaximumWidth(50)
        add_filter_btn.setMaximumHeight(22)
        add_filter_btn.clicked.connect(self._add_filter)
        add_filter_btn.setStyleSheet("font-size: 9px; padding: 2px;")
        filter_row.addWidget(add_filter_btn)

        # Show hidden instruments
        self.hidden_label = QLabel("Hidden: none")
        self.hidden_label.setStyleSheet("font-size: 8px; color: #999;")
        filter_row.addWidget(self.hidden_label)

        clear_filters_btn = QPushButton("Show All")
        clear_filters_btn.setMaximumWidth(60)
        clear_filters_btn.setMaximumHeight(22)
        clear_filters_btn.clicked.connect(self._clear_filters)
        clear_filters_btn.setStyleSheet("font-size: 9px; padding: 2px;")
        filter_row.addWidget(clear_filters_btn)

        header_layout.addLayout(filter_row)

        header.setLayout(header_layout)
        layout.addWidget(header)

        # --- Scroll area for flow cards ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Container for cards
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setAlignment(Qt.AlignTop)
        self.cards_layout.setSpacing(5)  # Compact spacing
        self.cards_layout.setContentsMargins(2, 2, 2, 2)

        scroll.setWidget(self.cards_container)
        layout.addWidget(scroll)

        # Info label
        self.info_label = QLabel("No flows detected yet")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        self.cards_layout.addWidget(self.info_label)

    def add_flow(self, flow_data: dict):
        """Add a new flow to monitor."""
        flow_id = flow_data['flow_id']
        instrument_id = flow_data.get('instrument_id', flow_data.get('ticker', 'N/A'))

        # Check if instrument is hidden
        if instrument_id in self.hidden_instruments:
            return

        if flow_id in self.flow_cards:
            # Update existing card
            self.flow_cards[flow_id].update_flow(flow_data)
        else:
            # Create new card
            card = FlowCard(flow_data)
            card.clicked.connect(self.flow_selected.emit)

            self.flow_cards[flow_id] = card

            # Hide info label if this is first flow
            if len(self.flow_cards) == 1:
                self.info_label.hide()

                # Auto-show widget when first flow arrives
                if self.auto_hide and not self.has_ever_had_flows:
                    self.show()
                    self.has_ever_had_flows = True

            # Add to layout
            self.cards_layout.addWidget(card)

        self._update_count()

    def update_flows(self, flows_data: List[dict]):
        """Update with list of flows."""
        for flow_data in flows_data:
            self.add_flow(flow_data)

    def remove_flow(self, flow_id: str):
        """Remove a flow from display."""
        if flow_id in self.flow_cards:
            card = self.flow_cards.pop(flow_id)
            self.cards_layout.removeWidget(card)
            card.deleteLater()

            # Show info label if no more flows
            if len(self.flow_cards) == 0:
                self.info_label.show()

            self._update_count()

    def _clear_completed(self):
        """Remove all completed (inactive) flows."""
        to_remove = []
        for flow_id, card in self.flow_cards.items():
            if not card.flow_data.get('is_active', True):
                to_remove.append(flow_id)

        for flow_id in to_remove:
            self.remove_flow(flow_id)

    def clear_all(self):
        """Clear ALL flows."""
        for flow_id in list(self.flow_cards.keys()):
            self.remove_flow(flow_id)

        # Reset visibility
        if self.auto_hide:
            self.hide()
            self.has_ever_had_flows = False

    def _add_filter(self):
        """Add instrument to hidden list."""
        instrument = self.filter_input.text().strip().upper()
        if instrument:
            self.hidden_instruments.add(instrument)
            self.filter_input.clear()
            self._update_hidden_label()
            self._apply_filters()

    def _clear_filters(self):
        """Clear all filters and show all instruments."""
        self.hidden_instruments.clear()
        self._update_hidden_label()
        self._apply_filters()

    def _apply_filters(self):
        """Reapply filters to all cards."""
        for flow_id, card in list(self.flow_cards.items()):
            instrument_id = card.flow_data.get('instrument_id',
                                               card.flow_data.get('ticker', 'N/A'))

            if instrument_id in self.hidden_instruments:
                # Hide card
                card.hide()
            else:
                # Show card
                card.show()

        self._update_count()

    def _update_hidden_label(self):
        """Update label showing hidden instruments."""
        if self.hidden_instruments:
            hidden_text = ", ".join(sorted(self.hidden_instruments))
            self.hidden_label.setText(f"Hidden: {hidden_text}")
        else:
            self.hidden_label.setText("Hidden: none")

    def _update_count(self):
        """Update count label."""
        visible_active = sum(1 for card in self.flow_cards.values()
                           if card.flow_data.get('is_active', True) and card.isVisible())
        visible_total = sum(1 for card in self.flow_cards.values() if card.isVisible())

        self.count_label.setText(f"Active: {visible_active} / {visible_total} shown")

    def get_active_flow_ids(self) -> List[str]:
        """Get list of active flow IDs."""
        return [flow_id for flow_id, card in self.flow_cards.items()
                if card.flow_data.get('is_active', True) and card.isVisible()]