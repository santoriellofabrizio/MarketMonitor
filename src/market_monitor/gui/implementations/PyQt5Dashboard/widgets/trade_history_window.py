"""
Trade History Window - Mostra tutti i trades per un ticker specifico.
Filtri: side, own trades, Min CTV.
OTTIMIZZATO: Rendering veloce di grandi dataset con QTableWidget
CON SUMMARY METRICS: Spread PL Sum e CTV Sum
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QLabel, QPushButton,
    QGroupBox, QCheckBox, QComboBox, QSpinBox,
    QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TradeHistoryWindow(QMainWindow):
    """Finestra per visualizzare la storia dei trades di un ticker."""

    def __init__(self, ticker: str, all_trades_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        
        logger.info(f"=== TradeHistoryWindow.__init__ START ===")
        logger.info(f"ticker: {ticker}")
        logger.info(f"all_trades_df shape: {all_trades_df.shape}")
        logger.info(f"all_trades_df columns: {list(all_trades_df.columns)}")
        
        self.ticker = ticker
        self.all_trades_df = all_trades_df.copy()
        self.filtered_trades_df = pd.DataFrame()
        self.current_filtered_df = pd.DataFrame()
        self.info_label = None  # Will be set in _setup_ui
        
        self.setWindowTitle(f"Trade History - {ticker}")
        self.setGeometry(100, 100, 1200, 700)
        
        try:
            logger.info("Calling _setup_ui...")
            self._setup_ui()
            logger.info("_setup_ui completed successfully")
            
            logger.info("Calling _load_trades...")
            self._load_trades()
            logger.info("_load_trades completed successfully")
            
            logger.info(f"=== TradeHistoryWindow.__init__ SUCCESS ===")
        except Exception as e:
            logger.error(f"CRITICAL ERROR in TradeHistoryWindow.__init__: {e}", exc_info=True)
            
            # If setup_ui failed, create minimal UI to show error
            if self.info_label is None:
                try:
                    central = QWidget()
                    self.setCentralWidget(central)
                    layout = QVBoxLayout(central)
                    self.info_label = QLabel()
                    layout.addWidget(self.info_label)
                except Exception as setup_err:
                    logger.error(f"Failed to create error UI: {setup_err}", exc_info=True)
                    raise
            
            # Display error
            self.info_label.setText(f"ERROR LOADING TRADES:\n{str(e)}")
            self.info_label.setStyleSheet("color: red; font-weight: bold; padding: 20px;")
            
            # Re-raise so caller knows something went wrong
            raise
        
    def _setup_ui(self):
        """Setup della UI."""
        logger.debug("_setup_ui: Creating central widget")
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # ========== METRICHE SUMMARY ==========
        logger.debug("_setup_ui: Creating summary metrics panel")
        metrics_group = QGroupBox("Summary Metrics")
        metrics_layout = QHBoxLayout()
        
        # Spread PL Sum
        self.spread_pl_label = QLabel("Spread PL Sum: 0.00")
        self.spread_pl_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        metrics_layout.addWidget(self.spread_pl_label)
        
        # CTV Sum
        self.ctv_sum_label = QLabel("CTV Sum: 0")
        self.ctv_sum_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        metrics_layout.addWidget(self.ctv_sum_label)
        
        metrics_layout.addStretch()
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # ========== FILTRI ==========
        logger.debug("_setup_ui: Creating filters group")
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        
        # Side filter
        filter_layout.addWidget(QLabel("Side:"))
        
        self.side_combo = QComboBox()
        self.side_combo.addItems(["All", "BUY", "SELL"])
        self.side_combo.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.side_combo)
        
        # Own trades checkbox
        self.own_trades_check = QCheckBox("Only Own Trades")
        self.own_trades_check.stateChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.own_trades_check)
        
        # Min CTV
        filter_layout.addWidget(QLabel("Min CTV:"))
        
        self.min_ctv = QSpinBox()
        self.min_ctv.setRange(0, 999999999)
        self.min_ctv.setSingleStep(1000)
        self.min_ctv.valueChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.min_ctv)
        
        filter_layout.addStretch()
        
        # Clear filters button
        clear_btn = QPushButton("Clear Filters")
        clear_btn.clicked.connect(self._clear_filters)
        filter_layout.addWidget(clear_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # ========== INFO LABEL ==========
        logger.debug("_setup_ui: Creating info label")
        self.info_label = QLabel("Loading trades...")
        self.info_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        layout.addWidget(self.info_label)
        
        # ========== TABELLA TRADES ==========
        logger.debug("_setup_ui: Creating table widget")
        self.table = QTableWidget()
        
        # OTTIMIZZAZIONI PER PERFORMANCE
        # 1. Disabilita sorting durante il popolamento
        self.table.setSortingEnabled(False)
        
        # 2. Usa SelectRows per selezione più veloce
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # 3. Disabilita alternating colors durante il caricamento (più veloce)
        self.table.setAlternatingRowColors(False)
        
        self.table.setColumnCount(0)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        
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
        logger.debug("_setup_ui: Completed successfully")
        
    def _load_trades(self):
        """Carica i trades del ticker."""
        logger.info(f"_load_trades: Starting for ticker '{self.ticker}'")
        logger.info(f"_load_trades: all_trades_df empty? {self.all_trades_df.empty}")
        logger.info(f"_load_trades: all_trades_df shape: {self.all_trades_df.shape}")
        
        if self.all_trades_df.empty:
            logger.warning("_load_trades: all_trades_df is empty!")
            self.info_label.setText("No trades available")
            return
        
        # Check if ticker column exists
        if 'ticker' not in self.all_trades_df.columns:
            logger.error(f"_load_trades: 'ticker' column not found! Available columns: {list(self.all_trades_df.columns)}")
            self.info_label.setText("ERROR: 'ticker' column not found in data")
            return
        
        logger.debug(f"_load_trades: Filtering for ticker '{self.ticker}'")
        logger.debug(f"_load_trades: Unique tickers in data: {self.all_trades_df['ticker'].unique().tolist()}")
        
        try:
            ticker_trades = self.all_trades_df[
                self.all_trades_df['ticker'].str.upper() == self.ticker.upper()
            ]
            logger.info(f"_load_trades: Found {len(ticker_trades)} trades for '{self.ticker}'")
        except Exception as e:
            logger.error(f"_load_trades: Error filtering trades: {e}", exc_info=True)
            self.info_label.setText(f"ERROR filtering trades: {e}")
            return
        
        if ticker_trades.empty:
            logger.warning(f"_load_trades: No trades found for ticker '{self.ticker}'")
            self.info_label.setText(f"No trades found for {self.ticker}")
            return
        
        self.filtered_trades_df = ticker_trades.copy()
        logger.info(f"_load_trades: Set filtered_trades_df with {len(self.filtered_trades_df)} rows")
        
        logger.info("_load_trades: Calling _apply_filters...")
        self._apply_filters()
        
    def _apply_filters(self):
        """Applica i filtri ai trades."""
        logger.debug(f"_apply_filters: Starting (filtered_trades_df has {len(self.filtered_trades_df)} rows)")
        
        try:
            df = self.filtered_trades_df.copy()
            
            if df.empty:
                logger.warning("_apply_filters: filtered_trades_df is empty after copy")
                self._populate_table(pd.DataFrame())
                self.info_label.setText(f"No trades found for {self.ticker}")
                self._update_metrics(pd.DataFrame())
                return
            
            logger.debug(f"_apply_filters: Starting with {len(df)} rows")
            
            # Filter per side
            side_filter = self.side_combo.currentText()
            if side_filter != "All":
                logger.debug(f"_apply_filters: Filtering by side '{side_filter}'")
                df = df[df['side'].str.upper() == side_filter.upper()]
                logger.debug(f"_apply_filters: After side filter: {len(df)} rows")
            
            # Filter per own trades
            if self.own_trades_check.isChecked():
                logger.debug("_apply_filters: Filtering for own trades only")
                df = df[df.get('own_trade', False) == True]
                logger.debug(f"_apply_filters: After own_trade filter: {len(df)} rows")
            
            # Filter per Min CTV
            if 'ctv' in df.columns:
                logger.debug("_apply_filters: Filtering by Min CTV")
                df = df[df['ctv'] >= self.min_ctv.value()]
                logger.debug(f"_apply_filters: After Min CTV filter: {len(df)} rows")
            elif 'quantity' in df.columns and 'price' in df.columns:
                logger.debug("_apply_filters: Calculating CTV from quantity * price")
                df['ctv'] = df['quantity'] * df['price']
                df = df[df['ctv'] >= self.min_ctv.value()]
                logger.debug(f"_apply_filters: After Min CTV filter: {len(df)} rows")
            
            # Sort per timestamp
            if 'timestamp' in df.columns:
                logger.debug("_apply_filters: Sorting by timestamp descending")
                df = df.sort_values('timestamp', ascending=False)
            
            logger.info(f"_apply_filters: Final result: {len(df)} rows")
            self.current_filtered_df = df.copy()
            self._populate_table(df)
            self._update_metrics(df)
            
            self.info_label.setText(
                f"{self.ticker} - {len(df)} trades (filtered from {len(self.filtered_trades_df)} total)"
            )
        except Exception as e:
            logger.error(f"Error applying filters: {e}", exc_info=True)
            self.info_label.setText(f"Error: {e}")
    
    def _update_metrics(self, df: pd.DataFrame):
        """Aggiorna le metriche di summary (spread_pl sum e ctv sum)."""
        logger.debug(f"_update_metrics: Starting with {len(df)} rows")
        
        if df.empty:
            self.spread_pl_label.setText("Spread PL Sum: 0.00")
            self.ctv_sum_label.setText("CTV Sum: 0")
            return
        
        # Calcola Spread PL Sum
        spread_pl_sum = 0.0
        if 'spread_pl' in df.columns:
            spread_pl_sum = df['spread_pl'].sum()
            logger.debug(f"_update_metrics: spread_pl_sum = {spread_pl_sum}")
        
        # Colorizza spread_pl_label in base al valore
        if spread_pl_sum >= 0:
            self.spread_pl_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px; color: green;")
        else:
            self.spread_pl_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px; color: red;")
        
        self.spread_pl_label.setText(f"Spread PL Sum: {spread_pl_sum:,.2f}")
        
        # Calcola CTV Sum
        ctv_sum = 0
        if 'ctv' in df.columns:
            ctv_sum = df['ctv'].sum()
            logger.debug(f"_update_metrics: ctv_sum = {ctv_sum}")
        elif 'quantity' in df.columns and 'price' in df.columns:
            ctv_sum = (df['quantity'] * df['price']).sum()
            logger.debug(f"_update_metrics: ctv_sum (calculated) = {ctv_sum}")
        
        self.ctv_sum_label.setText(f"CTV Sum: {ctv_sum:,.0f}")
        
        logger.debug(f"_update_metrics: Completed - spread_pl_sum={spread_pl_sum}, ctv_sum={ctv_sum}")
        
    def _populate_table(self, df: pd.DataFrame):
        """Popola la tabella con i trades - OTTIMIZZATO per performance."""
        logger.debug(f"_populate_table: Starting with {len(df)} rows")
        
        import time
        start_time = time.time()
        
        self.table.setRowCount(0)
        
        if df.empty:
            logger.debug("_populate_table: DataFrame is empty, clearing table")
            self.table.setColumnCount(0)
            return
        
        show_cols = [
            'timestamp', 'ticker', 'isin', 'side', 'quantity', 'price',
            'ctv', 'own_trade', 'spread_pl', 'market', 'currency'
        ]
        available_cols = [col for col in show_cols if col in df.columns]
        logger.info(f"_populate_table: Using columns: {available_cols}")
        
        # PRE-ALLOCATE: Imposta il numero di righe prima di popolare
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(available_cols))
        self.table.setHorizontalHeaderLabels(available_cols)
        
        # FORMATO: Pre-calcola i valori formattati FUORI da insertRow
        logger.debug("_populate_table: Pre-formatting values (outside loop)")
        
        # Preparazione dei dati
        formatted_data = []
        colors_info = []
        
        for row_idx, (_, row) in enumerate(df.iterrows()):
            row_data = []
            row_colors = []
            
            for col_name in available_cols:
                value = row[col_name]
                
                # Formattazione valore
                if col_name == 'timestamp':
                    text = str(value)[:19]
                elif col_name in ['price', 'spread_pl']:
                    text = f"{float(value):.4f}" if pd.notna(value) else "N/A"
                elif col_name == 'ctv':
                    text = f"{float(value):,.0f}" if pd.notna(value) else "N/A"
                elif col_name == 'quantity':
                    text = f"{int(value)}" if pd.notna(value) else "N/A"
                elif col_name == 'own_trade':
                    text = "✓" if value else ""
                elif col_name == 'side':
                    text = str(value).upper()
                else:
                    text = str(value)
                
                row_data.append(text)
                
                # Memorizza info colori
                color = None
                if col_name == 'side':
                    if text == 'BUY':
                        color = QColor(200, 255, 200)
                    elif text == 'SELL':
                        color = QColor(255, 200, 200)
                elif col_name == 'own_trade' and value:
                    color = QColor(200, 220, 255)
                elif col_name == 'spread_pl' and pd.notna(value):
                    val = float(value)
                    if val > 0:
                        color = QColor(200, 255, 200)
                    elif val < 0:
                        color = QColor(255, 200, 200)
                
                row_colors.append(color)
            
            formatted_data.append(row_data)
            colors_info.append(row_colors)
        
        # INSERISCI DATI: Ora popola la tabella CON BATCH
        logger.debug("_populate_table: Inserting data into table")
        
        for row_idx, (row_data, row_colors) in enumerate(zip(formatted_data, colors_info)):
            for col_idx, (text, color) in enumerate(zip(row_data, row_colors)):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                
                if color:
                    item.setBackground(color)
                
                self.table.setItem(row_idx, col_idx, item)
        
        # RESIZE COLUMNS: Dopo aver inserito tutto
        logger.debug("_populate_table: Resizing columns to fit content")
        for i in range(len(available_cols)):
            self.table.resizeColumnToContents(i)
            if i == 0:  # timestamp column
                self.table.setColumnWidth(i, 180)
        
        # ABILITA FEATURES: Alla fine
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        
        elapsed = time.time() - start_time
        logger.info(f"_populate_table: Completed successfully, showing {len(df)} rows in {elapsed:.3f}s")
    
    def _clear_filters(self):
        """Pulisce tutti i filtri."""
        logger.debug("_clear_filters: Resetting all filters")
        self.side_combo.setCurrentIndex(0)
        self.own_trades_check.setChecked(False)
        self.min_ctv.setValue(0)
        self._apply_filters()
