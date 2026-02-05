import logging
from typing import Dict

import numpy as np
import pandas as pd

from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.StrategyRegister import register_strategy


class SimplePriceMonitorStrategy(StrategyUI):
    """
    Strategia di test semplice che monitora i prezzi di mercato.

    Flusso:
    1. on_market_data_setting() -> configura subscription_dict
    2. wait_for_book_initialization() -> attende dati Bloomberg
    3. update_HF() -> legge prezzi da market_data.get_mid_eur()
    4. on_trade() -> stampa trades ricevuti

    Test: Verifica che i dati fluiscano correttamente
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("SimplePriceMonitorStrategy")

        # Configurazione strumenti
        self.isins = [
            "DE0007667107",  # Allianz
            "LU0048584102",  # Vanguard
            "LU0072462426",  # iShares
            "IE0002271879",  # Amundi
            "LU0073263215",  # Lyxor
        ]

        # Storage per monitoraggio
        self.price_history: Dict[str, list] = {isin: [] for isin in self.isins}
        self.trade_count = 0
        self.last_prices = {}

        self.logger.info(f"SimplePriceMonitorStrategy initialized with {len(self.isins)} instruments")

    def on_market_data_setting(self):
        """
        Configura il subscription dict per i strumenti da monitorare.
        Questo determina quali dati Bloomberg simulerà.
        """
        self.logger.info("Setting up market data subscriptions")

        # Crea subscription dict: ISIN -> "ISIN EQUITY"
        subscription_dict = {
            isin: f"{isin} EQUITY"
            for isin in self.isins
        }

        # Setta nel market_data (viene usato da BBGEventHandler)
        self.market_data.subscription_dict_bloomberg = subscription_dict
        self.market_data.securities = self.isins

        self.logger.info(f"Subscriptions set: {subscription_dict}")

    def wait_for_book_initialization(self) -> bool:
        """
        Attende che almeno uno strumento riceva dati.
        """
        self.logger.info("Waiting for book initialization...")

        # Attendi dati da almeno N strumenti
        mid = self.market_data.get_mid_eur()
        valid_data = mid[~mid.isna()]

        if len(valid_data) >= max(1, len(self.isins) // 2):
            self.logger.info(f"Book initialized with {len(valid_data)} instruments")
            return True

        return False

    def on_book_initialized(self):
        """Callback quando il book è inizializzato."""
        self.logger.info("Book initialized!")
        mid = self.market_data.get_mid_eur()
        self.logger.info(f"Initial prices:\n{mid[~mid.isna()].to_string()}")

    def update_HF(self):
        """
        Aggiornamento ad alta frequenza.
        Legge i prezzi correnti e li monitora.
        """
        try:
            # Leggi i prezzi correnti dal market data
            mid_eur = self.market_data.get_mid_eur()

            # Monitora i prezzi
            for isin in self.isins:
                if isin in mid_eur.index:
                    price = mid_eur[isin]
                    if not np.isnan(price) and price > 0:
                        self.price_history[isin].append(price)
                        self.last_prices[isin] = price

            # Log periodico ogni 10 aggiornamenti
            if len(self.price_history[self.isins[0]]) % 10 == 0:
                self._log_price_summary()

        except Exception as e:
            self.logger.error(f"Error in update_HF: {e}")

    def on_trade(self, trades: pd.DataFrame):
        """
        Callback quando arrivano trades di mercato.
        """
        try:
            self.trade_count += len(trades)

            self.logger.info(f"Received {len(trades)} trades (total: {self.trade_count})")

            # Log dei trades ricevuti
            for _, trade in trades.iterrows():
                self.logger.debug(
                    f"Trade: {trade.get('ticker', 'N/A')} | "
                    f"Qty: {trade.get('quantity', 0)} | "
                    f"Price: {trade.get('price', 0):.4f} | "
                    f"CTV: {trade.get('ctv', 0):.2f}"
                )

        except Exception as e:
            self.logger.error(f"Error in on_trade: {e}")

    def _log_price_summary(self):
        """Log summary dei prezzi correnti."""
        self.logger.info("=" * 60)
        for isin in self.isins:
            if isin in self.last_prices:
                price = self.last_prices[isin]
                updates = len(self.price_history[isin])
                self.logger.info(f"{isin}: {price:.4f} ({updates} updates)")
        self.logger.info("=" * 60)

    def on_stop(self):
        """Cleanup quando la strategia si ferma."""
        self.logger.info(f"Strategy stopped - Total trades: {self.trade_count}")

        # Report finale
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL REPORT")
        self.logger.info("=" * 60)
        for isin in self.isins:
            updates = len(self.price_history[isin])
            if updates > 0:
                prices = self.price_history[isin]
                self.logger.info(
                    f"{isin}: {updates} updates | "
                    f"Min: {min(prices):.4f} | "
                    f"Max: {max(prices):.4f} | "
                    f"Last: {prices[-1]:.4f}"
                )


register_strategy("SimplePriceMonitorStrategy", SimplePriceMonitorStrategy)